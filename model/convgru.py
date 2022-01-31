import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn import init

class ConvGRU(nn.Module):
    def __init__(self, channel_size):
        super(ConvGRU, self).__init__()
        self.c = channel_size
        self.Uz = nn.Conv2d(in_channels=self.c, out_channels=self.c, kernel_size=(3, 3), padding=1)
        self.Wz = nn.Conv2d(in_channels=self.c, out_channels=self.c, kernel_size=(3, 3), padding=1)
        self.Ur = nn.Conv2d(in_channels=self.c, out_channels=self.c, kernel_size=(3, 3), padding=1)
        self.Wr = nn.Conv2d(in_channels=self.c, out_channels=self.c, kernel_size=(3, 3), padding=1)
        self.W =  nn.Conv2d(in_channels=self.c, out_channels=self.c, kernel_size=(3, 3), padding=1)
        self.U =  nn.Conv2d(in_channels=self.c, out_channels=self.c, kernel_size=(3, 3), padding=1)

        init.kaiming_normal_(self.Ur.weight)
        init.kaiming_normal_(self.Wr.weight)
        init.kaiming_normal_(self.Uz.weight)
        init.kaiming_normal_(self.Wz.weight)
        init.kaiming_normal_(self.U.weight)
        init.kaiming_normal_(self.W.weight)
        init.constant_(self.Uz.bias, 0.)
        init.constant_(self.Ur.bias, 0.)
        init.constant_(self.U.bias, 0.)
        init.constant_(self.Wz.bias, 0.)
        init.constant_(self.Wr.bias, 0.)
        init.constant_(self.W.bias, 0.)


    def forward(self, message_in, node_hid):
        z = torch.sigmoid(self.Uz(node_hid) + self.Wz(message_in))
        r = torch.sigmoid(self.Ur(node_hid) + self.Wr(node_hid))
        candidate = torch.tanh(self.U(r*node_hid) + self.W(message_in))
        next_hidden = (1.0 - z)*node_hid + z*candidate
        return next_hidden



class ConvGRUCell(nn.Module):
    """
    Generate a convolutional GRU cell
    """

    def __init__(self, input_size, hidden_size, kernel_size):
        super().__init__()
        padding = kernel_size // 2
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.reset_gate = nn.Conv2d(input_size + hidden_size, hidden_size, kernel_size, padding=padding)
        self.update_gate = nn.Conv2d(input_size + hidden_size, hidden_size, kernel_size, padding=padding)
        self.out_gate = nn.Conv2d(input_size + hidden_size, hidden_size, kernel_size, padding=padding)
        self.dropout = nn.Dropout(p=0.5)
        self.ln1 = nn.LayerNorm([60, 60])
        self.ln2 = nn.LayerNorm([60, 60])
        self.ln3 = nn.LayerNorm([60, 60])

        init.orthogonal_(self.reset_gate.weight)
        init.orthogonal_(self.update_gate.weight)
        init.orthogonal_(self.out_gate.weight)
        init.constant_(self.reset_gate.bias, 0.)
        init.constant_(self.update_gate.bias, 0.)
        init.constant_(self.out_gate.bias, 0.)


    def forward(self, input_, prev_state):


        # data size is [batch, channel, height, width]

        stacked_inputs = torch.cat([input_, prev_state], dim=1)
        update = self.dropout(F.sigmoid(self.ln1(self.update_gate(stacked_inputs))))
        reset = self.dropout(F.sigmoid(self.ln2(self.reset_gate(stacked_inputs))))
        out_inputs = F.tanh(self.ln3(self.out_gate(torch.cat([input_, prev_state * reset], dim=1))))
        new_state = prev_state * (1 - update) + out_inputs * update

        return new_state


class ConvLSTMCell(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        """
        Initialize ConvLSTM cell.
        Parameters
        ----------
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))
