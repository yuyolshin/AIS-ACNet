import torch.optim as optim
from model import *
from geographic_utils import *
import utils


class trainer():
    def __init__(self, in_dim, seq_length, nhid, dropout, lrate, wdecay, device):
        self.model = gwnet(device, dropout,
                           in_dim=in_dim, out_dim=seq_length, residual_channels=nhid,
                           dilation_channels=nhid, skip_channels=nhid * 8, end_channels=nhid * 16)
        self.model.to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lrate, weight_decay=wdecay)
        self.loss = utils.haversine_loss
        self.clip = 5
        self.device = device
        self.alpha = 0.2
        self.beta = 0.2

    def train(self, input, real, ip_mask, op_mask, adj=None):

        self.model.train()
        self.optimizer.zero_grad()
        output, s_pred, h_pred = self.model(input, ip_mask, adj=adj)

        output = output.transpose(1, 3).contiguous()
        s_pred = s_pred.transpose(1, 3).contiguous().squeeze()
        h_pred = h_pred.transpose(1, 3).contiguous().squeeze()
        # output: [batch_size, t, num_nodes, 2]

        s_pred = s_pred * 51.1
        real[:, 2, :, :] = real[:, 2, :, :] * 51.1
        s_pred[~op_mask] = np.nan

        h_pred = h_pred * torch.pi
        real[:, 3, :, :] = real[:, 3, :, :] * torch.pi
        h_pred[~op_mask] = np.nan


        output[:, 0, :, :] = output[:, 0, :, :] * (max_lat - min_lat) + min_lat
        output[:, 1, :, :] = output[:, 1, :, :] * (max_lon - min_lon) + min_lon

        real[:, 0, :, :] = real[:, 0, :, :] * (max_lat - min_lat) + min_lat
        real[:, 1, :, :] = real[:, 1, :, :] * (max_lon - min_lon) + min_lon

        ADE = self.loss(real[:, :2, :, :], output, mask=op_mask)
        sloss = self.alpha * self.loss2(real[:, 2, :, :], s_pred, null_val=np.nan)
        hloss = self.beta * self.loss2(real[:, 3, :, :], h_pred, null_val=np.nan)
        loss = ADE + sloss + hloss

        loss.backward()
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()

        if op_mask is not None:
            fde = utils.haversine_loss(output[:, :, :, -1:], real[:, :2, :, -1:], mask=op_mask[:, :, -1:]).item()
        else:
            fde = utils.haversine_loss(output[:, :, :, -1:], real[:, :2, :, -1:], mask=None).item()

        if loss.item() == 0:
            print('loss = 0... Start running with verbose = 2')
            self.model(input, ip_mask, adj=adj, verbose=2)

        return loss.item(), ADE.item(), fde

    def eval(self, input, real, ip_mask, op_mask, adj=None, return_loss_terms=False):
        self.model.eval()
        output, s_pred, h_pred = self.model(input, ip_mask, adj=adj)
        output = output.transpose(1, 3).contiguous()
        s_pred = s_pred.transpose(1, 3).contiguous().squeeze()
        h_pred = h_pred.transpose(1, 3).contiguous().squeeze()
        # output = [batch_size, t, num_nodes, 2]

        s_pred = s_pred * 51.1
        real[:, 2, :, :] = real[:, 2, :, :] * 51.1
        s_pred[~op_mask] = np.nan

        h_pred = h_pred * torch.pi
        real[:, 3, :, :] = real[:, 3, :, :] * torch.pi
        h_pred[~op_mask] = np.nan

        output[:, 0, :, :] = output[:, 0, :, :] * (max_lat - min_lat) + min_lat
        output[:, 1, :, :] = output[:, 1, :, :] * (max_lon - min_lon) + min_lon

        real[:, 0, :, :] = real[:, 0, :, :] * (max_lat - min_lat) + min_lat
        real[:, 1, :, :] = real[:, 1, :, :] * (max_lon - min_lon) + min_lon

        ADE = self.loss(real[:, :2, :, :], output, mask=op_mask)
        sloss = self.alpha * self.loss2(real[:, 2, :, :], s_pred, null_val=np.nan)
        hloss = self.beta * self.loss2(real[:, 3, :, :], h_pred, null_val=np.nan)
        loss = ADE + sloss + hloss

        average_displacement = utils.haversine_loss(output, real, mask=op_mask, mse=True) # ADE

        if op_mask is not None:
            fde = utils.haversine_loss(output[:, :, :, -1:], real[:, :2, :, -1:], mask=op_mask[:, :, -1:]).item()
        else:
            fde = utils.haversine_loss(output[:, :, :, -1:], real[:, :2, :, -1:], mask=None).item()

        if return_loss_terms:
            return loss.item(), ADE.item(), fde, average_displacement.item(), sloss.item(), hloss.item()
        else:
            return loss.item(), ADE.item(), fde, average_displacement.item()
