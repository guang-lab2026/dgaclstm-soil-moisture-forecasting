
import torch
import torch.nn as nn
from torch.nn import Module, Sequential, Conv2d
from .predrnn import STLSTMCell

from functools import partial
from timm.models import named_apply
from .init_weights import _init_weights

class DiffH(nn.Module):
    def __init__(self, channels, kernel_size, forget_bias: float = 0.01):
        ""
        super().__init__()

        self.channels = channels
        self.forget_bias = forget_bias

        padding = (kernel_size // 2, kernel_size // 2)

        kernel_size = (kernel_size, kernel_size)

        self.conv_h_diff = nn.Conv2d(
            in_channels=channels, out_channels=channels * 4,
            kernel_size=kernel_size, padding=padding, stride=(1, 1)
        )

        self.conv_n = nn.Conv2d(
            in_channels=channels, out_channels=channels * 3,
            kernel_size=kernel_size, padding=padding, stride=(1, 1)
        )

        self.conv_w_no = nn.Conv2d(
            in_channels=channels, out_channels=channels,
            kernel_size=kernel_size, padding=padding, stride=(1, 1)
        )

    def forward(self, h_diff, n):
        ""
        h_diff_concat = self.conv_h_diff(h_diff)
        h_diff_concat = torch.layer_norm(h_diff_concat, h_diff_concat.shape[1:])
        n_concat = self.conv_n(n)
        n_concat = torch.layer_norm(n_concat, n_concat.shape[1:])

        g_h, i_h, f_h, o_h = torch.split(h_diff_concat, self.channels, dim=1)
        g_n, i_n, f_n = torch.split(n_concat, self.channels, dim=1)

        g = torch.tanh(g_h + g_n)
        i = torch.sigmoid(i_h + i_n)
        f = torch.sigmoid(f_h + f_n + self.forget_bias)

        new_n = f * n + i * g

        o_n = self.conv_w_no(new_n)
        o_n = torch.layer_norm(o_n, o_n.shape[1:])
        o = torch.sigmoid(o_h + o_n)
        diff = o * torch.tanh(new_n)

        return new_n, diff


class DCLSTMCell(Module):
    def __init__(self, in_channels: int, hidden_channels: int, kernel_size: int, forget_bias: float = 0.01):
        ""
        super().__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.forget_bias = forget_bias

        padding = (kernel_size // 2, kernel_size // 2)

        self.conv_x = nn.Conv2d(
            in_channels=in_channels, out_channels=hidden_channels * 7,
            kernel_size=(kernel_size, kernel_size), padding=padding, stride=(1, 1)
        )

        self.conv_h = nn.Conv2d(
            in_channels=hidden_channels, out_channels=hidden_channels * 4,
            kernel_size=(kernel_size, kernel_size), padding=padding, stride=(1, 1)
        )

        self.conv_m = nn.Conv2d(
            in_channels=hidden_channels, out_channels=hidden_channels * 3,
            kernel_size=(kernel_size, kernel_size), padding=padding, stride=(1, 1)
        )
        self.conv_c = nn.Conv2d(
            in_channels=hidden_channels * 2, out_channels=hidden_channels,
            kernel_size=(kernel_size, kernel_size), padding=padding, stride=(1, 1)
        )
        self.DiffH = DiffH(hidden_channels, kernel_size)

        self.conv_diffH = nn.Conv2d(
            in_channels=hidden_channels, out_channels=hidden_channels,
            kernel_size=(kernel_size, kernel_size), padding=padding, stride=(1, 1)
        )

        self.conv_o = nn.Conv2d(
            in_channels=hidden_channels * 2, out_channels=hidden_channels,
            kernel_size=(kernel_size, kernel_size), padding=padding, stride=(1, 1)
        )

        self.conv1x1 = nn.Conv2d(
            in_channels=hidden_channels * 2, out_channels=hidden_channels,
            kernel_size=(1, 1), padding=(0, 0), stride=(1, 1)
        )
        self.conv_hin = nn.Conv2d(
            in_channels=hidden_channels, out_channels=hidden_channels,
            kernel_size=(1, 1), padding=(0, 0), stride=(1, 1)
        )

    def forward(self, h_diff, x, c, h, m, h_DH):
        ""
        x_concat = self.conv_x(x)
        h_concat = self.conv_h(h)
        m_concat = self.conv_m(m)

        x_concat = torch.layer_norm(x_concat, x_concat.shape[1:])
        h_concat = torch.layer_norm(h_concat, h_concat.shape[1:])
        m_concat = torch.layer_norm(m_concat, m_concat.shape[1:])

        g_x, i_x, f_x, gg_x, ii_x, ff_x, o_x = torch.split(x_concat, self.hidden_channels, dim=1)
        g_h, i_h, o_h, f_h = torch.split(h_concat, self.hidden_channels, dim=1)
        gg_m, ii_m, ff_m = torch.split(m_concat, self.hidden_channels, dim=1)

        g = torch.tanh(g_x + g_h)
        i = torch.sigmoid(i_x + i_h)
        f = torch.sigmoid(f_x + f_h + self.forget_bias)

        h_DH, diff_h = self.DiffH(h_diff, h_DH)

        c = f * c + i * g
        c_n = self.conv_c(torch.cat([c, diff_h], dim=1))

        gg = torch.tanh(gg_x + gg_m)
        ii = torch.sigmoid(ii_x + ii_m)
        ff = torch.sigmoid(ff_x + ff_m + self.forget_bias)

        m = ff * m + ii * gg
        states = torch.cat([c_n, m], dim=1)

        o = torch.sigmoid(o_x + o_h + torch.tanh(self.conv_o(states)) + self.conv_diffH(diff_h))

        h_in = o * torch.tanh(self.conv1x1(states))
        h = self.conv_hin(h_in)

        return c_n, h, m, h_DH


class DCLSTM3(Module):
    def __init__(self, input_size, hidden_size: int, output_chans, filter_size, num_layers,
                 *args):
        super().__init__()
        self.n_layers = num_layers
        self.hidden_size = hidden_size
        # embedding layer
        self.embed = Conv2d(input_size, hidden_size, 1, 1, 0)

        # lstm layers
        lstm = []
        lstm.append(STLSTMCell(in_channels=hidden_size, hidden_channels=hidden_size, kernel_size=filter_size))
        for l in range(1, num_layers):
            lstm.append(DCLSTMCell(hidden_size, hidden_size, filter_size))
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
        h_last = []
        c = []
        h_DH = torch.zeros(batch, self.hidden_size, height, width).to(device)
        m = torch.zeros(batch, self.hidden_size, height, width).to(device)

        
        for i in range(self.n_layers):
            zero_tensor_h = torch.zeros(batch, self.hidden_size, height, width).to(device)
            c.append(zero_tensor_h)
            h.append(zero_tensor_h)
            h_last.append(zero_tensor_h)

        gen_ims = []
        for t in range(in_len + out_len):  # loop every seqs
            input_next = self.embed(inputs[:, t]) if t < in_len else last_time_output

            h[0], c[0], m = self.lstm[0](input_next, h[0], c[0], m)
            diff_h = h[0] - h_last[0]
            # loop subsequent layers
            for i in range(1, self.n_layers):
                c[i], h[i], m, h_DH = self.lstm[i](diff_h, h[i - 1], c[i], h[i], m, h_DH)
                diff_h = h[i] - h_last[i]

            h_last = [hi.clone() for hi in h]  # record current timestamp's glob
            last_time_output = h[self.n_layers - 1]

            gen_ims.append(self.output(last_time_output))

        prediction = torch.stack(gen_ims, dim=0)  # list to tensor
        return prediction, None


class DCLSTM31(Module):
    def __init__(self, input_size, hidden_size: int, output_chans, filter_size, num_layers,
                 *args):
        super().__init__()
        self.n_layers = num_layers
        self.hidden_size = hidden_size

        # embedding layer
        self.embed = Conv2d(input_size, hidden_size, 1, 1, 0)

        
        lstm = []

        for l in range(num_layers):
            lstm.append(DCLSTMCell(hidden_size, hidden_size, filter_size))
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

        
        h = [torch.zeros(batch, self.hidden_size, height, width, device=device)
             for _ in range(self.n_layers)]
        h_last = [torch.zeros(batch, self.hidden_size, height, width, device=device)
                  for _ in range(self.n_layers)]
        c = [torch.zeros(batch, self.hidden_size, height, width, device=device)
             for _ in range(self.n_layers)]
        h_DH = torch.zeros(batch, self.hidden_size, height, width, device=device)
        m = torch.zeros(batch, self.hidden_size, height, width, device=device)
        
        prev_input = torch.zeros(batch, self.hidden_size, height, width, device=device)

        gen_ims = []
        for t in range(in_len + out_len):  # loop every seqs
            input_next = self.embed(inputs[:, t]) if t < in_len else last_time_output

            
            diff_h = input_next - prev_input
            prev_input = input_next.clone()

            # c[0], h[0], m, h_DH = self.lstm[0](diff_h, input_next, c[0], h[0], m, h_DH)
            
            # for i in range(1, self.n_layers):
            #     diff_h = h[i - 1] - h_last[i - 1]
            #     c[i], h[i], m, h_DH = self.lstm[i](diff_h, h[i - 1], c[i], h[i], m, h_DH)

            for i in range(self.n_layers):
                c[i], h[i], m, h_DH = self.lstm[i](diff_h, input_next, c[i], h[i], m, h_DH)
                diff_h = h[i] - h_last[i]
                input_next = h[i]

                
            h_last = [hi.clone() for hi in h]
            last_time_output = h[self.n_layers - 1]
            gen_ims.append(self.output(last_time_output))

        prediction = torch.stack(gen_ims, dim=0)

        return prediction, None


class PredRNNDIFF(Module):
    def __init__(self, input_size, hidden_size: int, output_chans, filter_size, num_layers,
                 *args):
        super().__init__()
        self.n_layers = num_layers
        self.hidden_size = hidden_size

        # embedding layer
        self.embed = Conv2d(input_size, hidden_size, 1, 1, 0)

        
        lstm = []

        for l in range(num_layers):
            lstm.append(DCLSTMCell(hidden_size, hidden_size, filter_size))
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

        h = [torch.zeros(batch, self.hidden_size, height, width, device=device)
             for _ in range(self.n_layers)]
        h_last = [torch.zeros(batch, self.hidden_size, height, width, device=device)
                  for _ in range(self.n_layers)]
        c = [torch.zeros(batch, self.hidden_size, height, width, device=device)
             for _ in range(self.n_layers)]
        h_DH = [torch.zeros(batch, self.hidden_size, height, width, device=device)
                for _ in range(self.n_layers)]
        m = torch.zeros(batch, self.hidden_size, height, width, device=device)

        
        prev_input = torch.zeros(batch, self.hidden_size, height, width, device=device)

        gen_ims = []
        for t in range(in_len + out_len):  # loop every seqs
            input_next = self.embed(inputs[:, t]) if t < in_len else last_time_output

            
            diff_h = input_next - prev_input
            prev_input = input_next.clone()

            # c[0], h[0], m, h_DH[0] = self.lstm[0](diff_h, input_next, c[0], h[0], m, h_DH[0])
            
            # for i in range(1, self.n_layers):
            #     diff_h = h[i - 1] - h_last[i - 1]
            #     c[i], h[i], m, h_DH[i] = self.lstm[i](diff_h, h[i - 1], c[i], h[i], m, h_DH[i])

            for i in range(self.n_layers):
                c[i], h[i], m, h_DH[i] = self.lstm[i](diff_h, input_next, c[i], h[i], m, h_DH[i])
                diff_h = h[i] - h_last[i]
                input_next = h[i]

            
            h_last = [hi.clone() for hi in h]
            last_time_output = h[self.n_layers - 1]
            gen_ims.append(self.output(last_time_output))

        prediction = torch.stack(gen_ims, dim=0)

        return prediction, None
