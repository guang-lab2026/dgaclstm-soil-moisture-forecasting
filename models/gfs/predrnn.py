from typing import Tuple
import torch
from torch import nn, Tensor
from torch.nn import Module, Sequential, Conv2d
import torch.nn.functional as F

from functools import partial
from timm.models import named_apply
from .init_weights import _init_weights


class STLSTMCell(nn.Module):

    def __init__(self, in_channels: int, hidden_channels: int, kernel_size: int, forget_bias: float = 0.01):
        ""
        super().__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.forget_bias = forget_bias

        padding = (kernel_size // 2, kernel_size // 2)
        kernel_size = (kernel_size, kernel_size)

        
        self.conv_x = nn.Conv2d(
            in_channels=in_channels, out_channels=hidden_channels * 7,
            kernel_size=kernel_size, padding=padding, stride=(1, 1)
        )

        
        self.conv_h = nn.Conv2d(
            in_channels=hidden_channels, out_channels=hidden_channels * 4,
            kernel_size=kernel_size, padding=padding, stride=(1, 1)
        )

        
        self.conv_m = nn.Conv2d(
            in_channels=hidden_channels, out_channels=hidden_channels * 3,
            kernel_size=kernel_size, padding=padding, stride=(1, 1)
        )

        
        self.conv_o = nn.Conv2d(
            in_channels=hidden_channels * 2, out_channels=hidden_channels,
            kernel_size=kernel_size, padding=padding, stride=(1, 1)
        )

        
        self.conv1x1 = nn.Conv2d(
            in_channels=hidden_channels * 2, out_channels=hidden_channels,
            kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)
        )

    def forward(self, x: Tensor, h: Tensor, c: Tensor, m: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        ""
        if x is None and (h is None or c is None or m is None):
            raise ValueError("")

        x_concat = self.conv_x(x)
        h_concat = self.conv_h(h)
        m_concat = self.conv_m(m)

        x_concat = torch.layer_norm(x_concat, x_concat.shape[1:])
        h_concat = torch.layer_norm(h_concat, h_concat.shape[1:])
        m_concat = torch.layer_norm(m_concat, m_concat.shape[1:])

        g_x, i_x, f_x, gg_x, ii_x, ff_x, o_x = torch.split(x_concat, self.hidden_channels, dim=1)
        g_h, i_h, f_h, o_h = torch.split(h_concat, self.hidden_channels, dim=1)
        gg_m, ii_m, ff_m = torch.split(m_concat, self.hidden_channels, dim=1)

        g = torch.tanh(g_x + g_h)
        i = torch.sigmoid(i_x + i_h)
        f = torch.sigmoid(f_x + f_h + self.forget_bias)

        c = f * c + i * g

        gg = torch.tanh(gg_x + gg_m)
        ii = torch.sigmoid(ii_x + ii_m)
        ff = torch.sigmoid(ff_x + ff_m)

        m = ff * m + ii * gg

        states = torch.cat([c, m], dim=1)

        o = torch.sigmoid(o_x + o_h + self.conv_o(states))
        h = o * torch.tanh(self.conv1x1(states))

        return h, c, m


class PredRNN(nn.Module):
    def __init__(self, input_size, hidden_size: int, output_chans, filter_size, num_layers, forecast_chans=0, *args):
        super().__init__()
        self.n_layers = num_layers
        self.hidden_size = hidden_size
        self.forecast_chans = forecast_chans

        # embedding layer
        self.embed = Conv2d(input_size, hidden_size, 1, 1, 0)

        
        if forecast_chans > 0:
            self.forecast_fuse = Conv2d(hidden_size + forecast_chans, hidden_size, 1, 1, 0)
        else:
            raise ValueError("")
            # self.forecast_fuse = None

        # lstm layers
        lstm = []
        for l in range(0, num_layers):
            lstm.append(STLSTMCell(in_channels=hidden_size, hidden_channels=hidden_size, kernel_size=filter_size))

        self.lstm = nn.ModuleList(lstm)

        # output layer
        self.output = Conv2d(hidden_size, output_chans, 1, 1, 0)
        self.init_weights('normal')

    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)

    def forward(self, inputs, targets, other_forecast=None):
        device = inputs.device
        batch, in_len, feature_dim, height, width = inputs.shape
        _, out_len, _, _, _ = targets.shape

        m = torch.zeros(batch, self.hidden_size, height, width).to(device)

        h = [torch.zeros(batch, self.hidden_size, height, width, device=device) for _ in range(self.n_layers)]
        c = [torch.zeros(batch, self.hidden_size, height, width, device=device) for _ in range(self.n_layers)]

        gen_ims = []
        for t in range(in_len + out_len - 1):  # loop every seqs
            # input_next = self.embed(inputs[:, t]) if t < in_len else last_time_output
            if t < in_len:
                input_next = self.embed(inputs[:, t])
            else:
                
                input_next = last_time_output
                # if other_forecast is not None and self.forecast_fuse is not None:
                forecast_t = other_forecast[:, t - in_len]  # (batch, forecast_C, H, W)
                input_next = self.forecast_fuse(torch.cat([input_next, forecast_t], dim=1))

            for i in range(0, self.n_layers):
                h[i], c[i], m = self.lstm[i](input_next, h[i], c[i], m)
                input_next = h[i]

            last_time_output = h[self.n_layers - 1]

            gen_ims.append(self.output(last_time_output))

        prediction = torch.stack(gen_ims, dim=0)  # list to tensor
        return prediction, None


class STLSTMCellV2(nn.Module):

    def __init__(self, in_channels: int, hidden_channels: int, kernel_size: int, forget_bias: float = 0.01):
        ""
        super().__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.forget_bias = forget_bias

        padding = (kernel_size // 2, kernel_size // 2)
        kernel_size = (kernel_size, kernel_size)

        
        self.conv_x = nn.Conv2d(
            in_channels=in_channels, out_channels=hidden_channels * 7,
            kernel_size=kernel_size, padding=padding, stride=(1, 1)
        )

        
        self.conv_h = nn.Conv2d(
            in_channels=hidden_channels, out_channels=hidden_channels * 4,
            kernel_size=kernel_size, padding=padding, stride=(1, 1)
        )

        
        self.conv_m = nn.Conv2d(
            in_channels=hidden_channels, out_channels=hidden_channels * 3,
            kernel_size=kernel_size, padding=padding, stride=(1, 1)
        )

        
        self.conv_o = nn.Conv2d(
            in_channels=hidden_channels * 2, out_channels=hidden_channels,
            kernel_size=kernel_size, padding=padding, stride=(1, 1)
        )

        
        self.conv1x1 = nn.Conv2d(
            in_channels=hidden_channels * 2, out_channels=hidden_channels,
            kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)
        )

    def forward(self, x: Tensor, h: Tensor, c: Tensor, m: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        ""
        # if x is None and (h is None or c is None or m is None):
        

        x_concat = self.conv_x(x)
        h_concat = self.conv_h(h)
        m_concat = self.conv_m(m)

        x_concat = torch.layer_norm(x_concat, x_concat.shape[1:])
        h_concat = torch.layer_norm(h_concat, h_concat.shape[1:])
        m_concat = torch.layer_norm(m_concat, m_concat.shape[1:])

        g_x, i_x, f_x, gg_x, ii_x, ff_x, o_x = torch.split(x_concat, self.hidden_channels, dim=1)
        g_h, i_h, f_h, o_h = torch.split(h_concat, self.hidden_channels, dim=1)
        gg_m, ii_m, ff_m = torch.split(m_concat, self.hidden_channels, dim=1)

        g = torch.tanh(g_x + g_h)
        i = torch.sigmoid(i_x + i_h)
        f = torch.sigmoid(f_x + f_h + self.forget_bias)

        delta_c = g * i
        c = f * c + delta_c

        gg = torch.tanh(gg_x + gg_m)
        ii = torch.sigmoid(ii_x + ii_m)
        ff = torch.sigmoid(ff_x + ff_m)

        delta_m = ii * gg
        m = ff * m + delta_m

        states = torch.cat([c, m], dim=1)

        o = torch.sigmoid(o_x + o_h + self.conv_o(states))
        h = o * torch.tanh(self.conv1x1(states))

        return h, c, m, delta_c, delta_m


class PredRNNV2(nn.Module):
    def __init__(self, input_size, hidden_size: int, output_chans, filter_size, num_layers, forecast_chans, *args):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.forecast_chans = forecast_chans

        # embedding layer
        self.embed = Conv2d(input_size, hidden_size, 1, 1, 0)
        
        if forecast_chans > 0:
            self.forecast_fuse = Conv2d(hidden_size + forecast_chans, hidden_size, 1, 1, 0)
        else:
            raise ValueError("")
            # self.forecast_fuse = None

        # lstm layers
        lstm = []
        for l in range(0, num_layers):
            lstm.append(STLSTMCellV2(in_channels=hidden_size, hidden_channels=hidden_size, kernel_size=filter_size))

        self.lstm = nn.ModuleList(lstm)

        adapter_num_hidden = hidden_size
        self.adapter = nn.Conv2d(adapter_num_hidden, adapter_num_hidden, 1, stride=1, padding=0, bias=False)
        # output layer
        self.output = Conv2d(hidden_size, output_chans, 1, 1, 0)
        self.init_weights('normal')

    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)

    def forward(self, inputs, targets, other_forecast=None):
        device = inputs.device
        batch, in_len, feature_dim, height, width = inputs.shape
        _, out_len, _, _, _ = targets.shape

        h = []
        c = []
        delta_c_list = []
        delta_m_list = []
        m = torch.zeros(batch, self.hidden_size, height, width).to(device)

        
        for i in range(self.num_layers):
            zero_tensor_h = torch.zeros(batch, self.hidden_size, height, width).to(device)
            c.append(zero_tensor_h)
            h.append(zero_tensor_h)
            delta_c_list.append(zero_tensor_h)
            delta_m_list.append(zero_tensor_h)

        gen_ims = []
        decouple_loss = []

        for t in range(in_len + out_len):  # loop every seqs
            # input_next = self.embed(inputs[:, t]) if t < in_len else last_time_output
            if t < in_len:
                input_next = self.embed(inputs[:, t])
            else:
                
                input_next = last_time_output
                # if other_forecast is not None and self.forecast_fuse is not None:
                forecast_t = other_forecast[:, t - in_len]  # (batch,T, forecast_C, H, W)
                input_next = self.forecast_fuse(torch.cat([input_next, forecast_t], dim=1))

            for i in range(0, self.num_layers):
                h[i], c[i], m, delta_c, delta_m = self.lstm[i](input_next, h[i], c[i], m)
                delta_c_list[i] = F.normalize(self.adapter(delta_c).view(delta_c.shape[0], delta_c.shape[1], -1), dim=2)
                delta_m_list[i] = F.normalize(self.adapter(delta_m).view(delta_m.shape[0], delta_m.shape[1], -1), dim=2)
                input_next = h[i]

            last_time_output = h[self.num_layers - 1]
            gen_ims.append(self.output(last_time_output))

            for i in range(0, self.num_layers):
                decouple_loss.append(
                    torch.mean(torch.abs(torch.cosine_similarity(delta_c_list[i], delta_m_list[i], dim=2))))

        decouple_loss = torch.mean(torch.stack(decouple_loss, dim=0))
        prediction = torch.stack(gen_ims, dim=0)  # list to tensor
        return prediction, decouple_loss
