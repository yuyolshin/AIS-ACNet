from __future__ import print_function
import sys
import os
sys.dont_write_bytecode=True
import torch
import math
import pandas as pd
import numpy as np
import glob
from termcolor import colored
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from utils import *
from geographic_utils import *
import argparse


# pd.set_option('mode.chained_assignment',  None)


class trajectory_dataset(Dataset):
    def __init__(self, data_dir, sequence_length, prediction_length, feature_size):
        super(trajectory_dataset, self).__init__()
        self.filename = data_dir
        self.sequence_length = sequence_length
        self.prediction_length = prediction_length
        self.feature_size = feature_size
        self.len = 0
        self.sequences = {}
        self.masks = {}
        self.vesselCount = {}

        self.shift = self.sequence_length
        # for f, filename in enumerate(self.filenames):
        df = self.load_df(self.filename)
        df = self.normalize(df)
        if not df.empty:
            self.get_file_samples(df)

    def load_df(self, filename):
        df = pd.read_csv(filename, header=0, usecols=['BaseDateTime', 'MMSI', 'LAT', 'LON', 'SOG', 'COG'],
                         parse_dates=['BaseDateTime'])
        df.sort_values(['BaseDateTime'], inplace=True)
        return df

    def normalize(self, df):
        df['LAT'] = (math.pi/180)*df['LAT']
        df['LON'] = (math.pi/180)*df['LON']
        df = df.loc[(df['LAT'] <= max_lat) & (df['LAT'] >= min_lat) & (df['LON'] <= max_lon) & (df['LON'] >= min_lon)]
        if not df.empty:
            df['LAT'] = (df['LAT']-min_lat)/(max_lat-min_lat)
            df['LON'] = (df['LON']-min_lon)/(max_lon-min_lon) # min max norm
            # df['SOG'] = df['SOG']/22 # max speed?
            df['SOG'] = (df['SOG']-df['SOG'].min()) / (df['SOG'].max()-df['SOG'].min())
            df['COG'][df.COG < 0] = df['COG'].loc[df.COG < 0] + 360 + 49.6
            df['COG'] = df['COG'] / 360
            # df['Heading'] = df['Heading']/360 # max heading
            # print(df['SOG'].min(),df['SOG'].max())
        return df

    def get_file_samples(self, df):
        j = 0
        timestamps = df['BaseDateTime'].unique()
        # print("----processing datasets----")
        lents = len(timestamps)
        while not ((j+self.sequence_length+self.prediction_length) > lents):

            frame_timestamps = timestamps[j:j+self.sequence_length+self.prediction_length]
            frame = df.loc[df['BaseDateTime'].isin(frame_timestamps)]
            if self._condition_time(frame_timestamps):
                cond_val, vessels, valid_vessels = self._condition_vessels(frame)
                if cond_val:
                    frame = frame.loc[frame['MMSI'].isin(valid_vessels)]
                    total_valid_vessels = len(valid_vessels)
                    # sys.stdout.write(colored("\rfile: {}/{} Sample: {} Num Vessels: {}".format(f+1,len(self.filenames),self.len,vessels),"blue"))
                    sys.stdout.write(colored(
                        "\rSample: {} / {} Num Vessels: {}".format(self.len, lents, total_valid_vessels), "blue"))
                    self.sequences[self.len], self.masks[self.len], self.vesselCount[self.len] = self.get_sequence(frame)
                    self.len += 1
            # j+=self.shift
            j += 1

    def _condition_time(self, timestamps):
        # timesteps as pandas datetime
        condition_satisfied = True
        diff_timestamps = np.amax(np.diff(timestamps).astype('float'))
        if diff_timestamps / (6e+10) > 1 or diff_timestamps / (8.64e+13) >= 1:
            # discard the sequences with time step difference larger than 1 min or larger than 60 days
            condition_satisfied = False
        return condition_satisfied

    def _condition_vessels(self, frame):
        condition_satisfied = True
        # for train input
        frame_timestamps = frame['BaseDateTime'].unique()[:self.sequence_length]
        frame = frame.loc[frame['BaseDateTime'].isin(frame_timestamps)]

        total_vessels = len(frame['MMSI'].unique())
        # discard if the vessel is not moving
        valid_vessels = [v for v in frame['MMSI'].unique() \
                         if not abs(frame.loc[frame['MMSI'] == v]['LAT'].diff()).max() < (1e-04) \
                         and not abs(frame.loc[frame['MMSI'] == v]['LON'].diff()).max() < (1e-04) \
                         and len(frame.loc[frame['MMSI'] == v]) == self.sequence_length]

        # total vessel <= 3 discard
        # if (len(valid_vessels)<total_vessels) or total_vessels<=3:
        if len(valid_vessels) <= 3:
            condition_satisfied = False
        # return condition_satisfied, total_vessels
        return condition_satisfied, total_vessels, valid_vessels

    def get_sequence(self, frame):
        frame = frame.values
        frameIDs = np.unique(frame[:, 0]).tolist()
        input_frame = frame[np.isin(frame[:, 0], frameIDs[:self.sequence_length])]

        vessels = np.unique(input_frame[:, 1]).tolist()
        sequence = torch.FloatTensor(len(vessels), len(frameIDs), frame.shape[-1]-2)
        mask = torch.BoolTensor(len(vessels), len(frameIDs))

        for v, vessel in enumerate(vessels):
            vesselTraj = frame[frame[:, 1] == vessel]
            vesselTrajLen = np.shape(vesselTraj)[0]
            vesselIDs = np.unique(vesselTraj[:, 0])
            maskVessel = np.ones(len(frameIDs))

            if vesselTrajLen < (self.sequence_length + self.prediction_length):
                missingIDs = [f for f in frameIDs if not f in vesselIDs]
                maskVessel[vesselTrajLen:].fill(0.0)
                paddedTraj = np.zeros((len(missingIDs), np.shape(vesselTraj)[1]))
                vesselTraj=np.concatenate((vesselTraj, paddedTraj), axis=0)
                vesselTraj[vesselTrajLen:, 0] = missingIDs
                vesselTraj[vesselTrajLen:, 1] = vessel * np.ones((len(missingIDs)))
                sorted_idx = vesselTraj[:, 0].argsort()
                vesselTraj = vesselTraj[sorted_idx, :]
                if (np.max(vesselTraj[1:, 2] - vesselTraj[:-1, 2]) < 1e-04) and \
                        (np.max(vesselTraj[1:, 3] - vesselTraj[:-1, 3]) < 1e-04):
                    maskVessel[self.sequence_length:].fill(0.0)
                else:
                    maskVessel = maskVessel[sorted_idx]
                vesselTraj[:, 2:] = fillarr(vesselTraj[:, 2:])

            vesselTraj = vesselTraj[:, 2:]
            sequence[v, :] = torch.from_numpy(vesselTraj.astype('float32'))
            mask[v, :] = torch.from_numpy(maskVessel.astype('float32')).bool()
        vessel_count = torch.tensor(len(vessels))
        return sequence, mask, vessel_count, vessels

    def __getitem__(self, idx):
        idx = int(idx.numpy()) if not isinstance(idx, int) else idx

        sequence, mask, vessel_count= self.sequences[idx], self.masks[idx], self.vesselCount[idx]

        ip = sequence[:, :self.sequence_length, ...]
        op = sequence[:, self.sequence_length:, ...]
        ip_mask = mask[:, :self.sequence_length]
        op_mask = mask[:, self.sequence_length:]
        ip = ip[..., :self.feature_size]
        op = op[..., :self.feature_size]
        return {'input': ip,
                'output': op,
                'input_mask': ip_mask,
                'output_mask': op_mask,
                'vessels': vessel_count
                }

    def __len__(self):
        return self.len


def fillarr(arr):
    for i in range(arr.shape[1]):
        idx = np.arange(arr.shape[0])
        idx[arr[:, i] == 0] = 0
        np.maximum.accumulate(idx, axis=0, out=idx)
        if not np.sum(idx) == 0:
            arr[:, i] = arr[idx, i]
            if (arr[:, i] == 0).any():
                idx[arr[:, i] == 0] = 1e6
                idx = np.minimum.accumulate(idx[::-1], axis=0)[::-1]
                arr[:, i] = arr[idx, i]
    return arr


def pad_sequence(sequences, f, _len, padding_value=0.0):
    dim_ = sequences[0].size(1)
    if 'distance_matrix' in f:
        out_dims = (len(sequences), _len, dim_, _len)
    elif 'matrix' in f:
        out_dims = (len(sequences), _len, _len)
    elif 'mask' in f:
        out_dims = (len(sequences), _len, dim_)
    else:
        out_dims = (len(sequences), _len, dim_, sequences[0].size(-1))
    out_tensor = sequences[0].data.new(*out_dims).fill_(padding_value)
    for i, tensor in enumerate(sequences):
        length = tensor.size(0)
        if 'distance_matrix' in f:
            out_tensor[i, :length, :, :length] = tensor
        elif 'matrix' in f:
            out_tensor[i, :length, :length] = tensor
        else:
            out_tensor[i, :length, ...] = tensor
    return out_tensor


class collate_function:
    def __call__(self, batch):
        batch_size = len(batch)
        features = list(batch[0].keys())
        _len = max([b['input'].size(0) for b in batch])
        output_batch = []
        for f in features:
            if 'vessels' in f:
                output_feature = torch.stack([b[f].clone().detach().float() for b in batch])
            else:
                output_feature = pad_sequence([b[f] for b in batch], f, _len)
            output_batch.append(output_feature)
        return tuple(output_batch)


def load_data(args):

    train_dir = args.data_dir+'15/train/'
    val_dir = args.data_dir+'15/val/'
    test_dir = args.data_dir+'15/test/'
    if args.split_data or len(os.listdir(train_dir))==0:
        filename = args.data_dir + args.filename

        data = trajectory_dataset(filename, args.sequence_length, args.prediction_length, args.feature_size)
        data_size = len(data)

        valid_size = int(np.floor(0.1*data_size))
        test_size = int(np.floor(0.2*data_size))
        train_size = data_size-valid_size-test_size
        traindataset = Subset(data, range(train_size))
        validdataset = Subset(data, range(train_size, train_size+valid_size))
        testdataset = Subset(data, range(train_size+valid_size, train_size+valid_size+test_size))

        torch.save(traindataset, train_dir+"%02d_%02d.pt"%(args.sequence_length, args.prediction_length))
        torch.save(validdataset, val_dir+"%02d_%02d.pt"%(args.sequence_length, args.prediction_length))
        torch.save(testdataset, test_dir+"%02d_%02d.pt"%(args.sequence_length, args.prediction_length))
    else:
        traindataset = torch.load(train_dir+"%02d_%02d.pt"%(args.sequence_length,args.prediction_length))
        validdataset = torch.load(val_dir+"%02d_%02d.pt"%(args.sequence_length,args.prediction_length))
        testdataset = torch.load(test_dir+"%02d_%02d.pt"%(args.sequence_length,args.prediction_length))
    return traindataset, validdataset, testdataset


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default='/mnt/sda/AIS/processed_data/')
    parser.add_argument("--filename", type=str, default='15.csv')

    parser.add_argument("--split_data", action="store_false", help="split data into train, valid, test")
    parser.add_argument('--feature_size', type=int, default=4, help="feature size")
    parser.add_argument('--sequence_length', type=int, default=15, help="sequence length")
    parser.add_argument('--prediction_length', type=int, default=15, help="prediction length")

    args = parser.parse_args()

    load_data(args)

