import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import math
import sys


class SpatioTemporal_LSTM(nn.Module):
    """docstring for SpatioTemporal_LSTM"""
    def __init__(self, input_size, hidden_size, kernel_HW):
        super(SpatioTemporal_LSTM, self).__init__()
        self.input_size = input_size  # 3-tuple (iC, iH, hW)
        self.hidden_size = hidden_size  # 3-tuple (hC, hH, hW)
        self.kernel_HW = kernel_HW  # 2-tuple ((kH, kW)

        self.conv_xg = create_conv(self, 'ih')
        self.conv_xi = create_conv(self, 'ih')
        self.conv_xf = create_conv(self, 'ih')
        self.conv_hg = create_conv(self, 'hh')
        self.conv_hi =create_conv(self, 'hh')
        self.conv_hf =create_conv(self, 'hh')
        self.conv_cg =create_conv(self, 'hh')
        self.conv_ci =create_conv(self, 'hh')
        self.conv_cf =create_conv(self, 'hh')

        self.conv_xg_ = create_conv(self, 'ih')
        self.conv_xi_ = create_conv(self, 'ih')
        self.conv_xf_ = create_conv(self, 'ih')
        self.conv_cg_ = create_conv(self, 'hh')
        self.conv_ci_ = create_conv(self, 'hh')
        self.conv_cf_ = create_conv(self, 'hh')
        self.conv_mg_ = create_conv(self, 'ih')
        self.conv_mi_ = create_conv(self, 'ih')
        self.conv_mf_ = create_conv(self, 'ih')

        self.conv_mm = create_conv(self, 'ih')

        self.conv_xo = create_conv(self, 'ih')
        self.conv_co = create_conv(self, 'hh')
        self.conv_mo = create_conv(self, 'hh')

        self.conv_ch = create_conv(self, 'hh')
        self.conv_mh = create_conv(self, 'hh')


    def _compute_cell(self, x, h, c, m):
        # all parameters below is of (batch_size, oC, oH, oW)
        g = F.tanh(self.conv_xg(x) + self.conv_hg(h) + self.conv_cg(c))
        i = F.sigmoid(self.conv_xi(x) + self.conv_hi(h) + self.conv_ci(c))
        f = F.sigmoid(self.conv_xf(x) + self.conv_hf(h) + self.conv_cf(c))
        c = f * c + g * i

        g_ = F.tanh(self.conv_xg_(x) + self.conv_cg_(c) + self.conv_mg_(m))
        i_ = F.sigmoid(self.conv_xi_(x) + self.conv_ci(c) + self.conv_mi_(m))
        f_ = F.sigmoid(self.conv_xf_(x) + self.conv_cf(c) + self.conv_mf_(m))
        m = f_ * F.tanh(self.conv_mm(m)) + g_ * i_

        o = F.tanh(self.conv_xo(x) + self.conv_co(c) + self.conv_mo(m))
        h = o * F.tanh(self.conv_ch(c) + self.conv_mh(m))

        return h, c, m

    def forward(self, input_, state=None):
        if state is None:
            raise ValueError('empty initial state!')

        h, c, m = state

        cell_output = self._compute_cell(input_, h, c, m)

        return cell_output


class HighwayUnit(nn.Module):
    def __init__(self, input_size, hidden_size, kernel_HW):
        super(HighwayUnit, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.kernel_HW = kernel_HW

        self.conv_hp = create_conv(self, 'ih')
        self.conv_zp = create_conv(self, 'hh')
        self.conv_hu = create_conv(self, 'ih')
        self.conv_zu = create_conv(self, 'hh')


    def forward(self, input_, z=None):
        #input_ is the output of the lower unit, z is long term hidden state
        assert z is not None, 'empty state'
        p = F.tanh(self.conv_hp(input_) + self.conv_zp(z))  # information to add
        u = F.sigmoid(self.conv_hu(input_) + self.conv_zu(z))  # replace gate
        z = u * p + (1-u) * z
        return z


'''
helper functions
unit must have property input_size, hidden_size, kernel_HW, batch_size
'''
def create_conv(unit, kind):
    if kind == 'hh':
        return nn.Conv2d(unit.hidden_size[0], unit.hidden_size[0], unit.kernel_HW,
                        padding=((unit.kernel_HW[0]-1)/2, (unit.kernel_HW[1]-1)/2))
    elif kind == 'ih':
        return nn.Conv2d(unit.input_size[0], unit.hidden_size[0], unit.kernel_HW,
                         padding=((unit.hidden_size[1]-unit.input_size[1]+unit.kernel_HW[0]-1)/2,
                                 (unit.hidden_size[2]-unit.input_size[2]+unit.kernel_HW[1]-1)/2))
