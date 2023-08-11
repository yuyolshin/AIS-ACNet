from data import *
import argparse
from engine import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cuda:0', help='')
parser.add_argument('--train_dir', type=str, default='/mnt/sda/AIS/processed_data/15/train/15_15.pt')
parser.add_argument('--valid_dir', type=str, default='/mnt/sda/AIS/processed_data/15/val/15_15.pt')
parser.add_argument('--test_dir', type=str, default='/mnt/sda/AIS/processed_data/15/test/15_15.pt')
parser.add_argument('--mask_output', type=bool, default=True, help='whether to mask outputs for at anchor vessels')
parser.add_argument('--seq_length', type=int, default=15, help='')
parser.add_argument('--nhid', type=int, default=32, help='')
parser.add_argument('--in_dim', type=int, default=4, help='inputs dimension')
parser.add_argument('--batch_size', type=int, default=32, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
parser.add_argument('--dropout', type=float, default=0.3, help='dropout rate')
parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay rate')
parser.add_argument('--epochs', type=int, default=100, help='')
parser.add_argument('--print_every', type=int, default=50, help='')
# parser.add_argument('--seed', type=int, default=99, help='random seed')

parser.add_argument('--checkpoint',
                    default=['/mnt/sda/YS/AIS-PGCN/15_waveNet_dh_hd/best/exp_0_best_epoch_82_0.28.pth',
                             '/mnt/sda/YS/AIS-PGCN/15_waveNet_dh_hd/best/exp_1_best_epoch_73_0.27.pth',
                             '/mnt/sda/YS/AIS-PGCN/15_waveNet_dh_hd/best/exp_2_best_epoch_46_0.29.pth',
                             '/mnt/sda/YS/AIS-PGCN/15_waveNet_dh_hd/best/exp_3_best_epoch_98_0.28.pth',
                             '/mnt/sda/YS/AIS-PGCN/15_waveNet_dh_hd/best/exp_4_best_epoch_98_0.27.pth',
                             '/mnt/sda/YS/AIS-PGCN/15_waveNet_dh_hd/best/exp_5_best_epoch_92_0.29.pth'
                             ],
                    type=str, help='')
args = parser.parse_args()


def main(modelpath):
    device = torch.device(args.device)

    supports = None

    testdataset = torch.load(args.test_dir)

    testloader = DataLoader(testdataset, batch_size=args.batch_size, collate_fn=collate_function(), shuffle=False)


    outputs = []
    realy = []

    test_loss = []
    test_mde = []
    test_fde = []
    test_ade = []

    test_sloss = []
    test_hloss = []

    first_day_mde = []
    first_day_fde = []
    first_day_ade = []

    last_day_mde = []
    last_day_ade = []
    last_day_fde = []

    engine = trainer(args.in_dim, args.seq_length, args.nhid, args.dropout, args.learning_rate,
                     args.weight_decay, device, supports, args.gcn_bool, args.addaptadj, args.aptonly)
    engine.model.load_state_dict(torch.load(modelpath), strict=False)

    print('model load successfully')

    for iter, batch in enumerate(testloader):
        testx = batch[0].to(device)
        testx = testx.permute(0, 3, 1, 2)[:, :args.in_dim, ...]
        ip_mask = batch[-3].to(device)

        testy = batch[1].to(device)
        testy = testy.permute(0, 3, 1, 2)
        if args.mask_output:
            op_mask = batch[-2].to(device)
        else:
            op_mask = None

        with torch.no_grad():
            metrics = engine.eval(testx, testy, ip_mask, op_mask, adj=adj, return_loss_terms=True)

        with torch.no_grad():
            engine.model.eval()
            preds, _, _ = engine.model(testx, ip_mask, adj=adj)

        testy[:, 0, :, :] = testy[:, 0, :, :] * (max_lat - min_lat) + min_lat
        testy[:, 1, :, :] = testy[:, 1, :, :] * (max_lon - min_lon) + min_lon

        preds = preds.transpose(1, 3).contiguous()
        preds[:, 0, :, :] = preds[:, 0, :, :] * (max_lat - min_lat) + min_lat
        preds[:, 1, :, :] = preds[:, 1, :, :] * (max_lon - min_lon) + min_lon

        outputs.append(preds)
        realy.append(testy[:, :2, :, :])

        test_loss.append(metrics[0])
        test_mde.append(metrics[1])
        test_fde.append(metrics[2])
        test_ade.append(metrics[3])

        test_sloss.append(metrics[4])
        test_hloss.append(metrics[5])

        if iter < 45:
            first_day_mde.append(metrics[1])
            first_day_fde.append(metrics[2])
            first_day_ade.append(metrics[3])
        elif iter > (279 - 45):
            last_day_mde.append(metrics[1])
            last_day_fde.append(metrics[2])
            last_day_ade.append(metrics[3])

        testx[:, 0, :, :] = testx[:, 0, :, :] * (max_lat - min_lat) + min_lat
        testx[:, 1, :, :] = testx[:, 1, :, :] * (max_lon - min_lon) + min_lon

        np.save('/mnt/sda/YS/AIS-PGCN/15_waveNet/predictions/iter_{}_realx.npy'.format(int(iter)),
                testx.cpu().detach().numpy())
        np.save('/mnt/sda/YS/AIS-PGCN/15_waveNet/predictions/iter_{}_imask.npy'.format(int(iter)),
                ip_mask.cpu().detach().numpy())
        np.save('/mnt/sda/YS/AIS-PGCN/15_waveNet/predictions/iter_{}_realy.npy'.format(int(iter)),
                testy.cpu().detach().numpy())
        np.save('/mnt/sda/YS/AIS-PGCN/15_waveNet/predictions/iter_{}_preds.npy'.format(int(iter)),
                preds.cpu().detach().numpy())
        np.save('/mnt/sda/YS/AIS-PGCN/15_waveNet/predictions/iter_{}_masks.npy'.format(int(iter)),
                op_mask.cpu().detach().numpy())



    print(iter)
    mtest_loss = np.mean(test_loss)

    mtest_mde = np.mean(test_mde)

    mtest_fde = np.mean(test_fde)
    mtest_ade = np.mean(test_ade)

    mtest_sloss = np.mean(test_sloss)
    mtest_hloss = np.mean(test_hloss)

    mtest_fd_mde = np.mean(first_day_mde)
    mtest_fd_fde = np.mean(first_day_fde)
    mtest_fd_ade = np.mean(first_day_ade)

    mtest_ld_mde = np.mean(last_day_mde)
    mtest_ld_fde = np.mean(last_day_fde)
    mtest_ld_ade = np.mean(last_day_ade)

    log = 'On average over 15 horizons, Test Loss: {:.4f}, Test Mean displacement error: {:.4f}, ADE : {:.4f}, Test FDE: {:.4f}'
    print(log.format(mtest_loss, mtest_mde, mtest_ade, mtest_fde))

    log = 'On average over 15 horizons, Speed Loss: {:.4f}, Heading Loss: {:.4f}'
    print(log.format(mtest_sloss, mtest_hloss))

    log = 'On average over 15 horizons for the first day, Mean displacement error: {:.4f}, ADE : {:.4f}, FDE: {:.4f}'
    print(log.format(mtest_fd_mde, mtest_fd_ade, mtest_fd_fde))

    log = 'On average over 15 horizons for the last day, Mean displacement error: {:.4f}, ADE : {:.4f}, FDE: {:.4f}'
    print(log.format(mtest_ld_mde, mtest_ld_ade, mtest_ld_fde))

if __name__ == "__main__":
    for i in range(len(args.checkpoint)):
        main(args.checkpoint[i])
