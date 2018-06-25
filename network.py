from ST_LSTM import *
import sys

class Encoder(nn.Module):
    """
    n_layers is number of layers
    channels is number of channels of each layer. channels[0] is input channels, channel[n_layers] is output channels
    input_HW = (H, W), is height and width of input images
    kernel_HW is height and width of kernels used in convolutions
    """
    def __init__(self, input_size, hidden_sizes, kernel_HWs, kernel_HW_highway=None, kernel_HW_m=None, kernel_HW_output=None, encoder_only=True):
        super(Encoder, self).__init__()
        self.n_layers = len(hidden_sizes)  # at least 2
        self.input_size = input_size  # 3-tuple
        self.hidden_sizes = hidden_sizes  # list of 3-tuples of size self.n_layers
        self.kernel_HWs = kernel_HWs
        self.kernel_HW_highway = kernel_HWs[0] if kernel_HW_highway is None else kernel_HW_highway
        self.kernel_HW_m = kernel_HWs[0] if kernel_HW_m is None else kernel_HW_m
        self.kernel_HW_output = kernel_HW_output if kernel_HW_output is not None else kernel_HWs[-1]
        self.encoder_only = encoder_only


        self.cells = nn.ModuleList([])
        for i in range(self.n_layers):
            if i==0:
                cell = SpatioTemporal_LSTM(self.input_size, self.hidden_sizes[i], self.kernel_HWs[i])
            else:
                cell = SpatioTemporal_LSTM(self.hidden_sizes[i-1], self.hidden_sizes[i], self.kernel_HWs[i])
            self.cells.append(cell)

        self.highway_unit = HighwayUnit(self.hidden_sizes[0], self.hidden_sizes[0], self.kernel_HW_highway)

        self.resize_m_conv = nn.Conv2d(self.hidden_sizes[-1][0], self.input_size[0], self.kernel_HW_m,
                                       padding=((self.input_size[1]-self.hidden_sizes[-1][1]+self.kernel_HW_m[0]-1)/2,
                                                (self.input_size[2]-self.hidden_sizes[-1][2]+self.kernel_HW_m[1]-1)/2))
        if encoder_only:
            self.output_conv = nn.Conv2d(self.hidden_sizes[-1][0], self.input_size[0], self.kernel_HW_output,
                                    padding=((self.input_size[1]-self.hidden_sizes[-1][1]+self.kernel_HWs[-1][0]-1)/2,
                                             (self.input_size[2]-self.hidden_sizes[-1][2]+self.kernel_HWs[-1][1]-1)/2))

        self._reset_parameters()

    def _reset_parameters(self):
        for name, param in self.named_parameters():
            if len(param.size()) == 1:
                if name.find("f.") >= 0:
                    nn.init.constant_(param, 1)
                elif name.find("f_.") >= 0:
                    nn.init.constant_(param, 1)
                else:
                    nn.init.constant_(param, 0)
            elif len(param.size()) == 4:
                ran = 0.09/(param.size()[1]* param.size()[2]*param.size()[3])**0.3
                nn.init.uniform_(param, -ran, ran)
            else:
                print(len(param.size()))
                raise Exception('unexpected case')

    def forward(self, input_,  states=None):
        assert states is not None, "missing states"
        h, c, m, z = states
        for j, cell in enumerate(self.cells):  # j is layer
            if j == 0:
                resized_m = self.resize_m_conv(m[-1])
                h[j], c[j], m[j] = cell(input_, (h[j], c[j], resized_m))
            elif j==1:
                z = self.highway_unit(h[0], z)
                h[j], c[j], m[j] = cell(z, (h[j], c[j], m[j-1]))
            else:
                h[j], c[j], m[j] = cell(h[j-1], (h[j], c[j], m[j-1]))
        if not self.encoder_only:
            output = h[-1]
        else:
            output = self.output_conv(h[-1])
        return output, (h, c, m, z)






'''
class Decoder(nn.Module):
    """
    n_layers is number of ST_LSTM layers, which should be the same as the encoder
    input_channel is number of channels in the input
    context_channel is number of channels of context tensor
    channels is of length n_layers. channel[i] is the number of channels of ith ST_LSTM cell's hidden states
    input_HW is height and width of input and hidden states
    kernel_HW is the kernel used in performing convolution
    Decoder takes in a context and an input, outputs the prediction for next frame, then the context and
    last prediction was again put into the decoder.
    The internal states of decoder is initialized using final states of decoder.
    """
    def __init__(self, n_layers,  input_channel, context_channel, channels, input_HW, kernel_HW):
        super(Decoder, self).__init__()
        self.n_layers = n_layers
        self.input_channel = input_channel
        self.context_channel = context_channel
        self.channels = channels  # length n_layers+1
        self.input_HW = input_HW
        self.kernel_HW = kernel_HW

        stdv = 1.0 / math.sqrt(math.sqrt(self.input_HW[0]*self.input_HW[1]))
        self.weight_input = Parameter(torch.randn(self.channels[0], self.input_channel+self.context_channel, 1, 1)*stdv)
        self.bias_input = Parameter(torch.randn(self.channels[0], input_HW[0], input_HW[1]))
        self.weight_output = Parameter(torch.randn(self.input_channel, self.channels[n_layers-1], 1, 1)*stdv)
        self.bias_output = Parameter(torch.randn(self.input_channel, input_HW[0], input_HW[1]))

        self.cells = nn.ModuleList([])
        for i in range(self.n_layers):
            if i==0:
                cell = SpatioTemporal_LSTM(channels[i], channels[n_layers], channels[i+1], self.input_HW, self.kernel_HW)
            else:
                cell = SpatioTemporal_LSTM(channels[i], channels[i], channels[i+1], self.input_HW, self.kernel_HW)
            self.cells.append(cell)

    def forward(self, input_, context=None, states=None):
        print("forward called")
        assert states is not None, 'empty states for decoder'
        assert context is not None, 'empty states for decoder'
        input_and_context = F.tanh(F.conv2d(torch.cat((input_, context), dim=1), self.weight_input) + self.bias_input)
        h, c, m = states
        for j, cell in enumerate(self.cells):
            if j == 0:
                h[j], c[j], m[j] = cell(input_and_context, (h[j], c[j], m[self.n_layers-1]))
            else:
                h[j], c[j], m[j] = cell(h[j-1], (h[j], c[j], m[j-1]))

        return F.sigmoid(F.conv2d(h[self.n_layers-1], self.weight_output) + self.bias_output), (h, c, m)
'''
