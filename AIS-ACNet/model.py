import torch
import torch.nn as nn
import torch.nn.functional as F
# from util import pairwise_distances
import numpy as np
from torch.autograd import Variable
import sys


class linear(nn.Module):
    def __init__(self, c_in, c_out):
        super(linear, self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=True)

    def forward(self, x):
        return self.mlp(x)


class gwnet(nn.Module):
    def __init__(self, device, dropout=0.3, in_dim=2, out_dim=12, residual_channels=32, dilation_channels=32,
                 skip_channels=256, end_channels=512,
                 kernel_size=2, blocks=5, layers=2,):
        super(gwnet, self).__init__()
        self.dropout = dropout
        self.blocks = blocks
        self.layers = layers
        self.device = device

        """
        ModuleLists for the first encoder
        """
        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.fusion_convs = nn.ModuleList()
        self.fusion_convs_2 = nn.ModuleList()
        self.fusion_out1 = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.gconv = nn.ModuleList()

        self.start_conv = nn.Conv2d(in_channels=2,
                                    out_channels=residual_channels,
                                    kernel_size=(1, 1))

        """
        ModuleLists for the second encoder
        """
        self.filter_convs_a = nn.ModuleList()
        self.gate_convs_a = nn.ModuleList()
        self.residual_convs_a = nn.ModuleList()
        self.skip_convs_a = nn.ModuleList()
        self.fusion_convs_a = nn.ModuleList()
        self.fusion_convs_a_2 = nn.ModuleList()
        self.bn_a = nn.ModuleList()
        self.gconv_a = nn.ModuleList()

        self.start_conv_a = nn.Conv2d(in_channels=2,
                                      out_channels=residual_channels,
                                      kernel_size=(1, 1))

        receptive_field = 1

        for b in range(blocks):

            new_dilation = 1
            additional_scope = kernel_size - 1

            for i in range(layers):
                """
                For the first encoder
                """
                # dilated convolutions
                self.filter_convs.append(nn.Conv2d(in_channels=residual_channels,
                                                   out_channels=dilation_channels,
                                                   kernel_size=(1, kernel_size), dilation=new_dilation))

                self.gate_convs.append(nn.Conv2d(in_channels=residual_channels,
                                                 out_channels=dilation_channels,
                                                 kernel_size=(1, kernel_size), dilation=new_dilation))

                # 1x1 convolution for residual connection
                self.residual_convs.append(nn.Conv2d(in_channels=dilation_channels,
                                                     out_channels=residual_channels,
                                                     kernel_size=(1, 1)))

                self.fusion_convs.append(nn.Conv2d(in_channels=residual_channels,
                                                   out_channels=residual_channels,
                                                   kernel_size=(1, 1)))

                self.fusion_out1.append(nn.Conv2d(in_channels=residual_channels,
                                                  out_channels=residual_channels,
                                                  kernel_size=(1, 1)))

                self.bn.append(nn.BatchNorm2d(residual_channels))

                # 1x1 convolution for skip connection
                self.skip_convs.append(nn.Conv2d(in_channels=dilation_channels,
                                                 out_channels=skip_channels,
                                                 kernel_size=(1, 1)))
                """
                For the second encoder
                """

                self.filter_convs_a.append(nn.Conv2d(in_channels=residual_channels,
                                                     out_channels=dilation_channels,
                                                     kernel_size=(1, kernel_size), dilation=new_dilation))

                self.gate_convs_a.append(nn.Conv2d(in_channels=residual_channels,
                                                   out_channels=dilation_channels,
                                                   kernel_size=(1, kernel_size), dilation=new_dilation))

                # 1x1 convolution for residual connection
                self.residual_convs_a.append(nn.Conv2d(in_channels=dilation_channels,
                                                       out_channels=residual_channels,
                                                       kernel_size=(1, 1)))

                self.fusion_convs_a.append(nn.Conv2d(in_channels=residual_channels,
                                                     out_channels=residual_channels,
                                                     kernel_size=(1, 1)))


                self.skip_convs_a.append(nn.Conv2d(in_channels=dilation_channels,
                                                   out_channels=skip_channels,
                                                   kernel_size=(1, 1)))

                self.bn_a.append(nn.BatchNorm2d(residual_channels))
                new_dilation = 2
                receptive_field += additional_scope
                additional_scope = 2


        self.end_conv_1 = nn.Conv2d(in_channels=skip_channels,
                                    out_channels=end_channels,
                                    kernel_size=(1, 1),
                                    bias=True)

        self.end_conv_2x = nn.Conv2d(in_channels=end_channels,
                                     out_channels=out_dim,
                                     kernel_size=(1, 1),
                                     bias=True)

        self.end_conv_2y = nn.Conv2d(in_channels=end_channels,
                                     out_channels=out_dim,
                                     kernel_size=(1, 1),
                                     bias=True)


        self.end_conv_a_1 = nn.Conv2d(in_channels=skip_channels,
                                      out_channels=end_channels,
                                      kernel_size=(1, 1),
                                      bias=True)

        self.end_conv_a_s = nn.Conv2d(in_channels=end_channels,
                                      out_channels=out_dim,
                                      kernel_size=(1, 1),
                                      bias=True)

        self.end_conv_a_h = nn.Conv2d(in_channels=end_channels,
                                      out_channels=out_dim,
                                      kernel_size=(1, 1),
                                      bias=True)

        self.receptive_field = receptive_field

    def forward(self, input, ip_mask, adj=None, verbose=1):
        in_len = input.size(3)

        if in_len < self.receptive_field:
            x0 = nn.functional.pad(input, (self.receptive_field - in_len, 0, 0, 0))
        else:
            x0 = input

        x = self.start_conv(x0[:, :2, ...])
        x_a = self.start_conv_a(x0[:, 2:, ...])
        skip = 0
        skip_a = 0

        # WaveNet layers
        for i in range(self.blocks * self.layers):
            # print(torch.cuda.memory_allocated(device='cuda:0'))

            #            |----------------------------------------|     *residual*
            #            |                                        |
            #            |    |-- conv -- tanh --|                |
            # -> dilate -|----|                  * ----|-- 1x1 -- + -->	*input*
            #                 |-- conv -- sigm --|     |
            #                                         1x1
            #                                          |
            # ---------------------------------------> + ------------->	*skip*

            residual = x
            filter = self.filter_convs[i](residual)
            filter = torch.tanh(filter)
            gate = self.gate_convs[i](residual)
            gate = torch.sigmoid(gate)
            x = filter * gate

            # dilated convolution for the second encoder
            residual_a = x_a
            filter_a = self.filter_convs_a[i](residual_a)
            filter_a = torch.tanh(filter_a)
            gate_a = self.gate_convs_a[i](residual_a)
            gate_a = torch.sigmoid(gate_a)
            x_a = filter_a * gate_a

            # parametrized skip connection
            s = x
            s = self.skip_convs[i](s)

            s_a = x_a
            s_a = self.skip_convs_a[i](s_a)
            # method 1
            try:
                skip = skip[:, :, :, -s.size(3):]
                skip_a = skip_a[:, :, :, -s_a.size(3):]
            except:
                skip = 0
                skip_a = 0
            skip = s + skip
            skip_a = s_a + skip_a

            x = self.residual_convs[i](x)
            x_a = self.residual_convs_a[i](x_a)


            # feature fusion
            x_fusion = self.fusion_convs[i](x)
            x_a_fusion = self.fusion_convs_a[i](x_a)
            z = torch.sigmoid(torch.add(x_fusion, x_a_fusion))
            x = torch.add(torch.mul(z, x_fusion), torch.mul(1 - z, x_a_fusion))
            x = self.fusion_out1[i](x)
            x = x + residual[:, :, :, -x.size(3):]
            x_a = x_a + residual_a[:, :, :, -x_a.size(3):]

            x = self.bn[i](x)
            x_a = self.bn_a[i](x_a)

        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        x_lat = self.end_conv_2x(x)
        x_lon = self.end_conv_2y(x)

        x_a = F.relu(skip_a)
        x_a = F.relu(self.end_conv_a_1(x_a))
        x_s = self.end_conv_a_s(x_a)
        x_h = self.end_conv_a_h(x_a)

        x = torch.cat((x_lat, x_lon), dim=-1)

        ip_mask = ip_mask.permute(0, 2, 1).unsqueeze(-1)
        x = x * ip_mask
        x_s = x_s * ip_mask
        x_h = x_h * ip_mask

        return x, x_s, x_h





