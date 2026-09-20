import torch
import torch.nn as nn
import numpy as np
import torch.nn.init as init
from torch.nn import Module, Sequential, Conv2d
from functools import partial
from timm.models import named_apply
from .init_weights import _init_weights


class ConvLSTMCell(nn.Module):
    ""

    def __init__(self, in_channels, hidden_channels, kernel_size, forget_bias: float = 0.01):
        super().__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.forget_bias = forget_bias

        padding = (kernel_size // 2, kernel_size // 2)
        kernel_size = (kernel_size, kernel_size)

        
        self.conv_x = nn.Conv2d(in_channels=in_channels,
                                out_channels=4 * hidden_channels,
                                kernel_size=kernel_size,
                                padding=padding, stride=(1, 1))
        
        self.conv_h = nn.Conv2d(in_channels=hidden_channels,
                                out_channels=4 * hidden_channels,
                                kernel_size=kernel_size,
                                padding=padding, stride=(1, 1))

    def forward(self, input_tensor, h_cur, c_cur):
        ""

        
        conv_x = self.conv_x(input_tensor)  # [b, 4*hidden_dim, h, w]
        conv_h = self.conv_h(h_cur)  # [b, 4*hidden_dim, h, w]

        
        x_i, x_f, x_o, x_g = torch.split(conv_x, self.hidden_channels, dim=1)  # [b, hidden_dim, h, w]
        h_i, h_f, h_o, h_g = torch.split(conv_h, self.hidden_channels, dim=1)

        
        i = torch.sigmoid(x_i + h_i + c_cur)  # [b, hidden_dim, h, w]
        f = torch.sigmoid(x_f + h_f + c_cur + self.forget_bias)  # [b, hidden_dim, h, w]
        g = torch.tanh(x_g + h_g)  # [b, hidden_dim, h, w]
        o = torch.sigmoid(x_o + h_o + c_cur)  # [b, hidden_dim, h, w]

        
        c_next = f * c_cur + i * g  # [b, hidden_dim, h, w]
        h_next = o * torch.tanh(c_next)  # [b, hidden_dim, h, w]

        return h_next, c_next  # [b, output_dim, h, w] * 2


class ConvLSTM(nn.Module):
    def __init__(self, input_size, hidden_size: int, output_chans, filter_size, num_layers, *args):
        super().__init__()
        self.n_layers = num_layers
        self.hidden_size = hidden_size
        self.output_chans = output_chans
        # embedding layer
        self.embed = Conv2d(input_size, hidden_size, 1, 1, 0)

        # lstm layers
        lstm = []
        for l in range(0, num_layers):
            lstm.append(ConvLSTMCell(in_channels=hidden_size, hidden_channels=hidden_size, kernel_size=filter_size))

        self.lstm = nn.ModuleList(lstm)

        # output layer
        self.output = Conv2d(hidden_size, output_chans, 1, 1, 0)
        self.init_weights('normal')

    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)

    def forward(self, inputs, targets):
        device = inputs.device
        batch, in_len, feature_dim, height, width = inputs.shape
        _, out_len, _, _, _ = targets.shape

        h = []
        c = []

        
        for i in range(self.n_layers):
            zero_tensor_h = torch.zeros(batch, self.hidden_size, height, width).to(device)
            c.append(zero_tensor_h)
            h.append(zero_tensor_h)

        gen_ims = []
        for t in range(in_len + out_len - 1):  # loop every seqs
            input_next = self.embed(inputs[:, t]) if t < in_len else last_time_output

            for i in range(0, self.n_layers):
                h[i], c[i] = self.lstm[i](input_next, h[i], c[i])
                input_next = h[i]

            last_time_output = h[self.n_layers - 1]

            gen_ims.append(self.output(last_time_output))

        prediction = torch.stack(gen_ims, dim=0)  # list to tensor
        return prediction, None
