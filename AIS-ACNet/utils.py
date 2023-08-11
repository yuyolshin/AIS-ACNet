from __future__ import print_function
import sys
import os
sys.dont_write_bytecode=True
import torch
import numpy as np
import math
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import random
from termcolor import colored

from scipy.spatial.distance import pdist
from geographic_utils import *



seed = 10
seed = 300
seed - 10
def seed_everything(seed=100):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.initial_seed()
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything()


class Plotter(object):
    def __init__(self, args):
        super(Plotter, self).__init__()
        plot_dir = 'plots/'
        self.train_plots = plot_dir+str(args.model)+'/'+'hsz_'+str(args.hidden_size)+'_bsz_' + str(args.batch_size) + '_lr_'+str(args.learning_rate) + '_feature_size_'+str(args.feature_size)+'_sequence_length_'+str(args.sequence_length)+'_prediction_length_'+str(args.prediction_length)+'_delta_vals_'+str(args.delta_bearing)+'_'+str(args.delta_heading)+'_'+str(args.criterion_type)+'.png'
        if not os.path.isdir(plot_dir+str(args.model)):
            os.makedirs(plot_dir+str(args.model))
        print(colored("Saving learning curve at %s" %(self.train_plots),"blue"))
        self.train = []
        self.valid = []

    def update(self,train_loss,valid_loss):
        self.train.append(train_loss)
        self.valid.append(valid_loss)
        fig = plt.figure()
        plt.plot(self.train, 'r-', label='mean displacement error (training)')
        plt.plot(self.valid, 'b-',label='mean displacement error (validation)')
        plt.legend()
        plt.xlabel("Epoch")
        plt.savefig(self.train_plots)
        plt.close()


def haversine_loss(input_coords, pred_coords, null_val=np.nan, return_mean=True, mask=None, mse=False):
    # if np.isnan(null_val):
    #     mask = ~torch.isnan(input_coords)
    # else:
    #     mask = (input_coords != null_val)
    if mask is not None:
        mask = mask.float()
        mask /= torch.mean((mask))
        mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    else:
        mask = 1.0
    R = 3440.1
    # lat_errors = pred_coords[:, 0, :, :] - input_coords[:, 0, :, :]
    # lon_errors = pred_coords[:, 1, :, :] - input_coords[:, 1, :, :]
    lat1, lat2 = input_coords[:, 0, ...], pred_coords[:, 0, ...]
    lon1, lon2 = input_coords[:, 1, ...], pred_coords[:, 1, ...]

    # lat1, lon1 = scale_values(lat1, lon1)
    # lat2, lon2 = scale_values(lat2, lon2)

    lat_errors = lat2 - lat1
    lon_errors = lon2 - lon1
    a = torch.sin(lat_errors / 2) ** 2 \
        + torch.cos(input_coords[:, 0, :, :]) * torch.cos(pred_coords[:, 0, :, :]) * torch.sin(lon_errors / 2) ** 2
    a += 1e-24
    c = 2 * torch.atan2(torch.sqrt(a), torch.sqrt(1 - a))
    d = R * c

    d = d * mask
    d = torch.where(torch.isnan(d), torch.zeros_like(d), d)

    #     if torch.mean(d) == 0:
    #         np.save('pred.npy', pred_coords.cpu().detach().numpy())
    #         np.save('real.npy', input_coords.cpu().detach().numpy())

    # ade: d [B, f, N, T]
    # fde: d [B, N, T]

    if return_mean:
        if mse:
            return torch.mean(d ** 2)
        else:
            return torch.mean(d)
    else:
        return d


def distance_loss(x, y, mask=None, return_mean=True):
    if mask is not None:
        mask = mask.float()
        mask /= torch.mean((mask))
        mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    x = x.permute(0, 3, 2, 1)
    y = y.permute(0, 3, 2, 1)

    batch_size, time_stes, n_vessel, feature_size = x.size()
    x_norm = x.norm(dim=3)[:, :, :, None]
    y_t = y.permute(0, 1, 3, 2).contiguous()
    y_norm = y.norm(dim=3)[:, :, None]

    xy = torch.einsum('btnc, btcm->btnm', (x, y_t))

    dist = x_norm ** 2 + y_norm ** 2 - 2.0 * xy
    mask = mask.permute(0, 2, 1).unsqueeze(-1).expand_as(dist)
    dist = dist * mask
    dist = torch.where(torch.isnan(dist), torch.zeros_like(dist), dist)

    #     if torch.mean(d) == 0:
    #         np.save('pred.npy', pred_coords.cpu().detach().numpy())
    #         np.save('real.npy', input_coords.cpu().detach().numpy())

    # ade: d [B, f, N, T]
    # fde: d [B, N, T]

    if return_mean:
        return torch.mean(dist)
    else:
        return dist


def masked_mse(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = (preds-labels)**2
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def masked_rmse(preds, labels, null_val=np.nan):
    return torch.sqrt(masked_mse(preds=preds, labels=labels, null_val=null_val))


def masked_mae(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds-labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def masked_mape(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds-labels)/labels
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def metric(pred, real):
    mae = masked_mae(pred, real, 0.0).item()
    mape = masked_mape(pred, real, 0.0).item()
    rmse = masked_rmse(pred, real, 0.0).item()
    return mae, mape, rmse
