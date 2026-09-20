import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict
from .dgclstm import DGCLSTMCell

from functools import partial
from timm.models import named_apply
from .init_weights import _init_weights


# ==================== Temporal Attention ====================
class TemporalAttention(nn.Module):
    ""

    def __init__(self, hidden_channels: int, attention_dim: int = 64):
        super().__init__()
        self.hidden_channels = hidden_channels

        
        self.W_e = nn.Conv2d(hidden_channels * 2, attention_dim, 1)
        self.U_e = nn.Conv2d(hidden_channels, attention_dim, 1)
        self.V_e = nn.Conv2d(attention_dim, 1, 1)
        self.b_e = nn.Parameter(torch.zeros(1, attention_dim, 1, 1))

    def forward(self, encoder_outputs: torch.Tensor,
                decoder_h: torch.Tensor, decoder_c: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        ""
        B, T, C, H, W = encoder_outputs.shape

        
        decoder_concat = torch.cat([decoder_h, decoder_c], dim=1)
        decoder_term = self.W_e(decoder_concat)

        
        scores = []
        for k in range(T):
            h_k = encoder_outputs[:, k]
            score = self.V_e(torch.tanh(decoder_term + self.U_e(h_k) + self.b_e))
            score = score.mean(dim=[2, 3])
            scores.append(score)

        scores = torch.cat(scores, dim=1)
        attention_weights = F.softmax(scores, dim=1)

        
        attention_weights_expanded = attention_weights.view(B, T, 1, 1, 1)
        context = (encoder_outputs * attention_weights_expanded).sum(dim=1)

        return context, attention_weights


# ==================== Encoder ====================
class DGCLSTMEncoder(nn.Module):
    ""

    def __init__(self, input_channels: int, hidden_channels: int, kernel_size: int,
                 num_layers: int = 3, fixed_number: int = 0):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_channels = hidden_channels
        self.fixed_number = fixed_number

        
        self.embed = nn.Conv2d(input_channels - fixed_number, hidden_channels, 1)

        
        if fixed_number > 0:
            padding_ = kernel_size // 2
            self.G_info = nn.Sequential(
                nn.Conv2d(fixed_number, hidden_channels, kernel_size, 1, padding_),
                nn.BatchNorm2d(hidden_channels),
                nn.ReLU(),
                nn.Conv2d(hidden_channels, hidden_channels * 2, kernel_size, 1, padding_),
                nn.BatchNorm2d(hidden_channels * 2),
                nn.Sigmoid(),
            )

        
        self.layers = nn.ModuleList([
            DGCLSTMCell(hidden_channels, hidden_channels, kernel_size)
            for _ in range(num_layers)
        ])
        self.init_weights('normal')

    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        ""
        B, T, C, H, W = x.shape
        device = x.device










        
        h = [torch.zeros(B, self.hidden_channels, H, W, device=device) for _ in range(self.num_layers)]
        h_last = [torch.zeros(B, self.hidden_channels, H, W, device=device) for _ in range(self.num_layers)]
        c = [torch.zeros(B, self.hidden_channels, H, W, device=device) for _ in range(self.num_layers)]
        h_DH = [torch.zeros(B, self.hidden_channels, H, W, device=device)
                for _ in range(self.num_layers)]
        m = torch.zeros(B, self.hidden_channels, H, W, device=device)

        prev_input = torch.zeros(B, self.hidden_channels, H, W, device=device)

        
        if self.fixed_number > 0:
            G_info = self.G_info(x[:, 0, -self.fixed_number:])
            G_H = torch.split(G_info, self.hidden_channels, dim=1)
        x = x[:, :, :-self.fixed_number]
        outputs = []

        for t in range(T):
            input_next = self.embed(x[:, t])

            
            diff_h = input_next - prev_input
            prev_input = input_next.clone()

            
            # c[0], h[0], m, h_DH = self.layers[0](diff_input, input_t, c[0], h[0], m, h_DH, G_H)
            # diff_h = h[0] - h_last[0]
            #
            
            # for i in range(1, self.num_layers):
            #     c[i], h[i], m, h_DH = self.layers[i](diff_h, h[i - 1], c[i], h[i], m, h_DH, G_H)
            #     diff_h = h[i] - h_last[i]




            for i in range(self.num_layers):
                c[i], h[i], m, h_DH[i] = self.layers[i](diff_h, input_next, c[i], h[i], m, h_DH[i], G_H)
                diff_h = h[i] - h_last[i]
                input_next = h[i]

            h_last = [hi.clone() for hi in h]
            outputs.append(h[self.num_layers - 1])

        outputs = torch.stack(outputs, dim=1)

        states = {
            'h': h,
            'c': c,
            'm': m,
            'h_DH': h_DH,
            'h_last': h_last,
            'G_H': G_H,
            'prev_output': h[self.num_layers - 1]
        }

        return outputs, states


# ==================== Decoder ====================
class DGCLSTMDecoder(nn.Module):
    ""

    def __init__(self, input_channels: int, hidden_channels: int, output_channels: int,
                 kernel_size: int, num_layers: int = 3):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_channels = hidden_channels

        
        self.layers = nn.ModuleList([
            DGCLSTMCell(hidden_channels, hidden_channels, kernel_size)
            for _ in range(num_layers)
        ])

        
        self.input_proj = nn.Conv2d(input_channels, hidden_channels, 1)

        
        self.output_conv = nn.Conv2d(hidden_channels, output_channels, 1)

    def forward(self, x: torch.Tensor, states: Dict) -> Tuple[torch.Tensor, Dict]:
        ""
        h = states['h']
        c = states['c']
        m = states['m']
        h_DH = states['h_DH']
        h_last = states['h_last']
        G_H = states['G_H']
        prev_output = states['prev_output']

        
        input_t = self.input_proj(x)

        
        diff_h = input_t - prev_output

        
        # c[0], h[0], m, h_DH = self.layers[0](diff_input, input_t, c[0], h[0], m, h_DH, G_H)
        # diff_h = h[0] - h_last[0]
        #
        
        # for i in range(1, self.num_layers):
        #     c[i], h[i], m, h_DH = self.layers[i](diff_h, h[i - 1], c[i], h[i], m, h_DH, G_H)
        #     diff_h = h[i] - h_last[i]




        for i in range(self.num_layers):
            c[i], h[i], m, h_DH[i] = self.layers[i](diff_h, input_t, c[i], h[i], m, h_DH[i], G_H)
            diff_h = h[i] - h_last[i]
            input_t = h[i]

        h_last = [hi.clone() for hi in h]

        output = self.output_conv(h[-1])

        new_states = {
            'h': h,
            'c': c,
            'm': m,
            'h_DH': h_DH,
            'h_last': h_last,
            'G_H': G_H,
            'prev_output': h[-1]
        }

        return output, new_states


# ==================== ATedDGCLSTM ====================
class DGACLSTM(nn.Module):
    ""

    def __init__(self, input_channels: int, hidden_channels: int, output_channels: int,
                 kernel_size: int = 3, num_layers: int = 3, *args):
        super().__init__()
        attention_dim = hidden_channels * 4
        self.hidden_channels = hidden_channels
        self.output_channels = output_channels
        self.output_chans = output_channels
        fixed_number = args[0]["fixed_number"]
        self.fixed_number = fixed_number

        
        self.encoder = DGCLSTMEncoder(input_channels, hidden_channels, kernel_size,
                                      num_layers, fixed_number)

        
        self.attention = TemporalAttention(hidden_channels, attention_dim)

        
        self.decoder = DGCLSTMDecoder(hidden_channels * 2, hidden_channels,
                                      output_channels, kernel_size, num_layers)

        
        self.start_conv = nn.Conv2d(hidden_channels, hidden_channels, 1)
        self.output_conv = nn.Conv2d(hidden_channels, output_channels, 1)

        self.init_weights('normal')

    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor = None
                ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        ""
        B, T_in, C, H, W = inputs.shape
        T_out = targets.shape[1] if targets is not None else T_in

        
        encoder_outputs, encoder_states = self.encoder(inputs)

        
        decoder_states = encoder_states.copy()

        
        predictions = []
        for t in range(T_in):
            predictions.append(self.output_conv(encoder_outputs[:, t]))

        last_output = self.start_conv(encoder_states['h'][-1])
        decoder_states['prev_output'] = last_output

        
        for t in range(T_out):
            
            context, attn_weights = self.attention(encoder_outputs, decoder_states['h'][0],
                                                   decoder_states['c'][0])

            decoder_input = torch.cat([context, last_output], dim=1)

            
            output, decoder_states = self.decoder(decoder_input, decoder_states)
            last_output = decoder_states['h'][-1]
            predictions.append(output)

        predictions = torch.stack(predictions, dim=0)

        return predictions, None


# ==================== EdDGCLSTM ====================
class EncoderDecoderNoAttention(nn.Module):
    ""

    def __init__(self, input_channels: int, hidden_channels: int, output_channels: int,
                 kernel_size: int = 3, num_layers: int = 3, *args):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.output_channels = output_channels
        self.output_chans = output_channels
        fixed_number = args[0]["fixed_number"]
        self.fixed_number = fixed_number

        
        self.encoder = DGCLSTMEncoder(input_channels, hidden_channels, kernel_size,
                                      num_layers, fixed_number)

        
        self.decoder = DGCLSTMDecoder(hidden_channels, hidden_channels,
                                      output_channels, kernel_size, num_layers)

        
        self.start_conv = nn.Conv2d(hidden_channels, hidden_channels, 1)
        self.output_conv = nn.Conv2d(hidden_channels, output_channels, 1)

        self.init_weights('normal')

    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor = None
                ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        ""
        B, T_in, C, H, W = inputs.shape
        T_out = targets.shape[1] if targets is not None else T_in

        
        encoder_outputs, encoder_states = self.encoder(inputs)

        
        decoder_states = encoder_states.copy()

        
        predictions = []
        for t in range(T_in):
            predictions.append(self.output_conv(encoder_outputs[:, t]))

        last_output = self.start_conv(encoder_states['h'][-1])
        decoder_states['prev_output'] = last_output

        
        for t in range(T_out):
            output, decoder_states = self.decoder(last_output, decoder_states)
            last_output = decoder_states['h'][-1]
            predictions.append(output)

        predictions = torch.stack(predictions, dim=0)

        return predictions, None

