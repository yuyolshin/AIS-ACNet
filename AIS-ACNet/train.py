import torch
import numpy as np
import argparse
import time
import pickle
from data import *
import matplotlib.pyplot as plt
from engine import trainer
from torch.utils.data import DataLoader

import os
import sys


parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cuda:1', help='')
parser.add_argument('--train_dir', type=str, default='/mnt/sda/AIS/processed_data/15/train/15_15.pt')
parser.add_argument('--valid_dir', type=str, default='/mnt/sda/AIS/processed_data/15/val/15_15.pt')
parser.add_argument('--test_dir', type=str, default='/mnt/sda/AIS/processed_data/15/test/15_15.pt')
parser.add_argument('--save', type=str, default='/mnt/sda/YS/AIS-PGCN/15_wavenet_dsh_200epochs/',
                    help='save path')
parser.add_argument('--mask_output', type=bool, default=True, help='whether to mask outputs for at anchor vessels')
parser.add_argument('--seq_length', type=int, default=15, help='')
parser.add_argument('--nhid', type=int, default=32, help='')
parser.add_argument('--in_dim', type=int, default=4, help='inputs dimension')
parser.add_argument('--batch_size', type=int, default=32, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
parser.add_argument('--dropout', type=float, default=0.3, help='dropout rate')
parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay rate')
parser.add_argument('--epochs', type=int, default=200, help='')
parser.add_argument('--print_every', type=int, default=50, help='')
# parser.add_argument('--seed', type=int, default=99, help='random seed')

args = parser.parse_args()


def main(expid=0):
    device = torch.device(args.device)

    traindataset = torch.load(args.train_dir)
    validdataset = torch.load(args.valid_dir)
    testdataset = torch.load(args.test_dir)

    trainloader = DataLoader(traindataset, batch_size=args.batch_size, collate_fn=collate_function(), shuffle=True)
    validloader = DataLoader(validdataset, batch_size=args.batch_size, collate_fn=collate_function(), shuffle=True)
    print(args)

    engine = trainer(args.in_dim, args.seq_length, args.nhid, args.dropout, args.learning_rate,
                     args.weight_decay, device, supports, args.gcn_bool, args.addaptadj, args.aptonly)
    print(sum(p.numel() for p in engine.model.parameters() if p.requires_grad))

    print("start training...", flush=True)
    his_tr_loss = []
    his_loss = []
    val_time = []
    train_time = []
    for i in range(1, args.epochs + 1):
        train_loss = []
        train_mape = []
        train_rmse = []
        t1 = time.time()
        for iter, batch in enumerate(trainloader):

            trainx = batch[0].to(device)
            trainx = trainx.permute(0, 3, 1, 2)[:, :args.in_dim, ...]
            ip_mask = batch[-3].to(device)

            trainy = batch[1].to(device)
            trainy = trainy.permute(0, 3, 1, 2)
            if args.mask_output:
                op_mask = batch[-2].to(device)
            else:
                op_mask = None

            metrics = engine.train(trainx, trainy, ip_mask, op_mask, adj=adj)
            train_loss.append(metrics[0])
            train_mape.append(metrics[1])
            train_rmse.append(metrics[2])
            if metrics[0] == 0:
                print(
                    'Iter: {:03d}, loss: {:.4f}, mask sum: {:.2f}'.format(iter, metrics[0], torch.sum(op_mask).item()))

            if iter % args.print_every == 0:
                log = 'Iter: {:03d}, Train Loss: {:.4f}, Train ADE: {:.4f}, Train FDE: {:.4f}'
                print(log.format(iter, train_loss[-1], train_mape[-1], train_rmse[-1]), flush=True)

        t2 = time.time()
        train_time.append(t2 - t1)

        valid_loss = []
        valid_mape = []
        valid_rmse = []

        s1 = time.time()
        for iter, batch in enumerate(validloader):
            testx = batch[0].to(device)
            testx = testx.permute(0, 3, 1, 2)[:, :args.in_dim, ...]
            ip_mask = batch[-3].to(device)

            testy = batch[1].to(device)
            testy = testy.permute(0, 3, 1, 2)
            if args.mask_output:
                op_mask = batch[-2].to(device)
            else:
                op_mask = None

            metrics = engine.eval(testx, testy, ip_mask, op_mask, adj=adj)
            valid_loss.append(metrics[0])
            valid_mape.append(metrics[1])
            valid_rmse.append(metrics[2])

        s2 = time.time()
        log = 'Epoch: {:03d}, Inference Time: {:.4f} secs'
        print(log.format(i, (s2 - s1)))
        val_time.append(s2 - s1)
        mtrain_loss = np.mean(train_loss)
        mtrain_mape = np.mean(train_mape)
        mtrain_rmse = np.mean(train_rmse)
        his_tr_loss.append(mtrain_loss)

        mvalid_loss = np.mean(valid_loss)
        mvalid_mape = np.mean(valid_mape)
        mvalid_rmse = np.mean(valid_rmse)
        his_loss.append(mvalid_loss)

        log = 'Epoch: {:03d}, Train Loss: {:.4f}, Train ADE: {:.4f}, Train FDE: {:.4f}, Valid Loss: {:.4f}, Valid ADE: {:.4f}, Valid FDE: {:.4f}, Training Time: {:.4f}/epoch'
        print(log.format(i, mtrain_loss, mtrain_mape, mtrain_rmse, mvalid_loss, mvalid_mape, mvalid_rmse, (t2 - t1)),
              flush=True)
        torch.save(engine.model.state_dict(),
                   args.save + "exp_" + str(expid) + "_epoch_" + str(i) + "_" + str(round(mvalid_loss, 2)) + ".pth")
    print("Average Training Time: {:.4f} secs/epoch".format(np.mean(train_time)))
    print("Average Inference Time: {:.4f} secs".format(np.mean(val_time)))

    # testing
    bestid = np.argmin(his_loss)
    engine.model.load_state_dict(
        torch.load(args.save + "exp_" + str(expid) + "_epoch_" + str(bestid + 1) + "_" + str(
            round(his_loss[bestid], 2)) + ".pth"))
    outputs = []
    realy = []
    testloader = DataLoader(testdataset, batch_size=args.batch_size, collate_fn=collate_function(), shuffle=False)

    test_loss = []
    test_ade = []
    test_fde = []

    engine.model.eval()

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
            preds, s_pred, h_pred = engine.model(testx, ip_mask, adj=adj)
        preds = preds.transpose(1, 3)
        outputs.append(preds)
        realy.append(testy)

        metrics = engine.eval(testx, testy, ip_mask, op_mask, adj=adj)
        test_loss.append(metrics[0])
        test_ade.append(metrics[1])
        test_fde.append(metrics[2])

    mtest_loss = np.mean(test_loss)
    mtest_ade = np.mean(test_ade)
    mtest_fde = np.mean(test_fde)

    print("Training finished")

    log = 'On average over 15 horizons, Test Loss: {:.4f}, Test ADE: {:.4f}, Test FDE: {:.4f}'
    print(log.format(mtest_loss, mtest_ade, mtest_fde))
    print('best epoch: ', str(bestid))
    torch.save(engine.model.state_dict(),
               args.save + 'best/exp_' + str(expid) + '_best_epoch_' + str(bestid) +
               '_' + str(round(his_loss[bestid], 2)) + ".pth")

    np.savetxt(args.save + 'best/train_loss_' + str(expid) + '_' + str(round(his_tr_loss[bestid], 2)) + '.txt', his_tr_loss)
    np.savetxt(args.save + 'best/valid_loss_' + str(expid) + '_' + str(round(his_loss[bestid], 2)) + '.txt', his_loss)


if __name__ == "__main__":
    t1 = time.time()
    main()
    t2 = time.time()
    print("Total time spent: {:.4f}".format(t2 - t1))
