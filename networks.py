"""
In this module a bunch of network components are defined as well as some wrappers into 
bigger networks.

Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
from torch import nn
from torch.autograd import Variable
import torch
import torch.nn.functional as F
from torch.nn.modules.utils import _quadruple
from torch.nn.common_types import _size_4_t


##################################################################################
# Discriminator
##################################################################################

class MsImageDis(nn.Module):
    # Multi-scale discriminator architecture
    def __init__(self, input_dim, land_mask, params):
        super(MsImageDis, self).__init__()
        self.n_layer = params['n_layer']
        self.gan_type = params['gan_type']
        self.dim = params['dim']
        self.norm = params['norm']
        self.activ = params['activ']
        self.num_scales = params['num_scales']
        self.pad_type = params['pad_type']
        self.input_dim = input_dim
        self.use_land_mask = land_mask is not None
        self.land_mask = nn.Parameter(land_mask, requires_grad=False)
        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)
        self.cnns = nn.ModuleList()
        for _ in range(self.num_scales):
            self.cnns.append(self._make_net())

    def _make_net(self):
        dim = self.dim
        cnn_x = []
        cnn_x += [
            Conv2dBlock(
                (self.input_dim+int(self.use_land_mask)), dim, 4, 2, 1, 
                norm='none', 
                activation=self.activ, 
                pad_type=self.pad_type)
        ]
        for i in range(self.n_layer - 1):
            cnn_x += [Conv2dBlock(dim, dim * 2, 4, 2, 1, norm=self.norm, activation=self.activ, pad_type=self.pad_type)]
            dim *= 2
        cnn_x += [nn.Conv2d(dim, 1, 1, 1, 0)]
        cnn_x = nn.Sequential(*cnn_x)
        return cnn_x

    def forward(self, x):
        outputs = []
        if self.use_land_mask:
            x = torch.cat([x, self.land_mask.clone().detach().repeat(x.shape[0],1,1,1)], dim=1)
        for model in self.cnns:
            outputs.append(model(x))
            x = self.downsample(x)
        return outputs

    def calc_dis_loss(self, input_fake, input_real):
        # calculate the loss to train D
        outs0 = self.forward(input_fake)
        outs1 = self.forward(input_real)
        loss = 0

        for it, (out0, out1) in enumerate(zip(outs0, outs1)):
            if self.gan_type == 'lsgan':
                loss += torch.mean((out0 - 0)**2) + torch.mean((out1 - 1)**2)
            elif self.gan_type == 'nsgan':
                all0 = Variable(torch.zeros_like(out0.data).cuda(), requires_grad=False)
                all1 = Variable(torch.ones_like(out1.data).cuda(), requires_grad=False)
                loss += torch.mean(F.binary_cross_entropy(F.sigmoid(out0), all0) +
                                   F.binary_cross_entropy(F.sigmoid(out1), all1))
            else:
                assert 0, "Unsupported GAN type: {}".format(self.gan_type)
        return loss

    def calc_gen_loss(self, input_fake):
        # calculate the loss to train G
        outs0 = self.forward(input_fake)
        loss = 0
        for it, (out0) in enumerate(outs0):
            if self.gan_type == 'lsgan':
                loss += torch.mean((out0 - 1)**2) # LSGAN
            elif self.gan_type == 'nsgan':
                all1 = Variable(torch.ones_like(out0.data).cuda(), requires_grad=False)
                loss += torch.mean(F.binary_cross_entropy(F.sigmoid(out0), all1))
            else:
                assert 0, "Unsupported GAN type: {}".format(self.gan_type)
        return loss


##################################################################################
# Generator
##################################################################################

class VAEGen(nn.Module):
    # VAE architecture
    def __init__(self, input_dim, land_mask, params):
        super(VAEGen, self).__init__()
        dim = params['dim']
        n_downsample = params['n_downsample']
        n_res = params['n_res']
        activ = params['activ']
        pad_type = params['pad_type']
        output_activ = params['output_activ']
        upsample = params['upsample']
        self.use_land_mask = land_mask is not None
        self.land_mask = nn.Parameter(land_mask, requires_grad=False)

        # content encoder
        self.enc = ContentEncoder(n_downsample, n_res, input_dim+int(self.use_land_mask), dim, 'in', activ, pad_type=pad_type)
        self.dec = Decoder(n_downsample, n_res, self.enc.output_dim, input_dim, res_norm='in', 
                           activ=activ, pad_type=pad_type, output_activ=output_activ, upsample=upsample)

    def forward(self, images):
        # This is a reduced VAE implementation where we assume the outputs are multivariate Gaussian distribution with mean = hiddens and std_dev = all ones.
        hiddens = self.encode(images)
        if self.training == True:
            noise = Variable(torch.randn(hiddens.size()).cuda(hiddens.data.get_device()))
            images_recon = self.decode(hiddens + noise)
        else:
            images_recon = self.decode(hiddens)
        return images_recon, hiddens

    def encode(self, images):
        if self.use_land_mask:
            images = torch.cat([images, self.land_mask.clone().detach().repeat(images.shape[0],1,1,1)], dim=1)
        hiddens = self.enc(images)
        noise = Variable(torch.randn(hiddens.size()).cuda(hiddens.data.get_device()))
        return hiddens, noise

    def decode(self, hiddens):
        images = self.dec(hiddens)
        return images


##################################################################################
# Encoder and Decoders
##################################################################################


class ContentEncoder(nn.Module):
    def __init__(self, n_downsample, n_res, input_dim, dim, norm, activ, pad_type):
        super(ContentEncoder, self).__init__()
        self.model = []
        self.model += [Conv2dBlock(input_dim, dim, 7, 1, 3, norm=norm, activation=activ, pad_type=pad_type)]
        # downsampling blocks
        for i in range(n_downsample):
            self.model += [Conv2dBlock(dim, 2 * dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)]
            dim *= 2
        # residual blocks
        self.model += [ResBlocks(n_res, dim, norm=norm, activation=activ, pad_type=pad_type)]
        self.model = nn.Sequential(*self.model)
        self.output_dim = dim

    def forward(self, x):
        return self.model(x)


class Decoder(nn.Module):
    def __init__(self, n_upsample, n_res, dim, output_dim, res_norm='adain', 
                 activ='relu', pad_type='zero', output_activ='none', upsample='nearest'):
        super(Decoder, self).__init__()
        
        if upsample=='nearest':
            def _decode_upsampler(C): return nn.Upsample(scale_factor=2)
        elif upsample=='bilinear':
            def _decode_upsampler(C): return nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        elif upsample=='conv': 
            def _decode_upsampler(C): return nn.ConvTranspose2d(C, C, kernel_size=2, stride=2)
        else:
            assert 0, "`upsample` must be one of ['nearest', 'bilinear', 'conv']"

        self.model = []
        # AdaIN residual blocks
        self.model += [ResBlocks(n_res, dim, res_norm, activ, pad_type=pad_type)]
        # upsampling blocks
        for i in range(n_upsample):
            self.model += [_decode_upsampler(dim),
                           Conv2dBlock(dim, dim // 2, 5, 1, 2, norm='ln', activation=activ, pad_type=pad_type)]
            dim //= 2

        self.model += [Conv2dBlock(dim, output_dim, 7, 1, 3, norm='none', activation=output_activ, pad_type=pad_type)]
        self.model = nn.Sequential(*self.model)


    def forward(self, x):
        return self.model(x)


##################################################################################
# Sequential Models
##################################################################################

class ResBlocks(nn.Module):
    def __init__(self, num_blocks, dim, norm='in', activation='relu', pad_type='zero'):
        super(ResBlocks, self).__init__()
        self.model = []
        for i in range(num_blocks):
            self.model += [ResBlock(dim, norm=norm, activation=activation, pad_type=pad_type)]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x)


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, dim, n_blk, norm='none', activ='relu'):

        super(MLP, self).__init__()
        self.model = []
        self.model += [LinearBlock(input_dim, dim, norm=norm, activation=activ)]
        for i in range(n_blk - 2):
            self.model += [LinearBlock(dim, dim, norm=norm, activation=activ)]
        self.model += [LinearBlock(dim, output_dim, norm='none', activation='none')] # no output activations
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x.view(x.size(0), -1))


##################################################################################
# Basic Blocks
##################################################################################

class ResBlock(nn.Module):
    def __init__(self, dim, norm='in', activation='relu', pad_type='zero'):
        super(ResBlock, self).__init__()

        model = []
        model += [Conv2dBlock(dim ,dim, 3, 1, 1, norm=norm, activation=activation, pad_type=pad_type)]
        model += [Conv2dBlock(dim ,dim, 3, 1, 1, norm=norm, activation='none', pad_type=pad_type)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        residual = x
        out = self.model(x)
        out += residual
        return out


class Conv2dBlock(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, stride,
                 padding=0, norm='none', activation='relu', pad_type='zero'):
        super(Conv2dBlock, self).__init__()
        self.use_bias = True
        # initialize padding
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        elif pad_type == 'lonwrap':
            self.pad = LatLonPad(padding, lon_dim=-1)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)

        # initialize normalization
        norm_dim = output_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(norm_dim)
        elif norm == 'in':
            #self.norm = nn.InstanceNorm2d(norm_dim, track_running_stats=True)
            self.norm = nn.InstanceNorm2d(norm_dim)
        elif norm == 'ln':
            self.norm = LayerNorm(norm_dim)
        elif norm == 'adain':
            self.norm = AdaptiveInstanceNorm2d(norm_dim)
        elif norm == 'none':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        self.activation = MultiChannelActivation(activation, output_dim)

        # initialize convolution
        self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride, bias=self.use_bias)

    def forward(self, x):
        x = self.conv(self.pad(x))
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x


class LinearBlock(nn.Module):
    def __init__(self, input_dim, output_dim, norm='none', activation='relu'):
        super(LinearBlock, self).__init__()
        use_bias = True
        # initialize fully connected layer
        self.fc = nn.Linear(input_dim, output_dim, bias=use_bias)

        # initialize normalization
        norm_dim = output_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm1d(norm_dim)
        elif norm == 'in':
            self.norm = nn.InstanceNorm1d(norm_dim)
        elif norm == 'ln':
            self.norm = LayerNorm(norm_dim)
        elif norm == 'none':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        self.activation = get_activation(activation)


    def forward(self, x):
        out = self.fc(x)
        if self.norm:
            out = self.norm(out)
        if self.activation:
            out = self.activation(out)
        return out


##################################################################################
# Normalization layers
##################################################################################

class AdaptiveInstanceNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(AdaptiveInstanceNorm2d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        # weight and bias are dynamically assigned
        self.weight = None
        self.bias = None
        # just dummy buffers, not used
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

    def forward(self, x):
        assert self.weight is not None and self.bias is not None, "Please assign weight and bias before calling AdaIN!"
        b, c = x.size(0), x.size(1)
        running_mean = self.running_mean.repeat(b)
        running_var = self.running_var.repeat(b)

        # Apply instance norm
        x_reshaped = x.contiguous().view(1, b * c, *x.size()[2:])

        out = F.batch_norm(
            x_reshaped, running_mean, running_var, self.weight, self.bias,
            True, self.momentum, self.eps)

        return out.view(b, c, *x.size()[2:])

    def __repr__(self):
        return self.__class__.__name__ + '(' + str(self.num_features) + ')'


class LayerNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True):
        super(LayerNorm, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps

        if self.affine:
            self.gamma = nn.Parameter(torch.Tensor(num_features).uniform_())
            self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        shape = [-1] + [1] * (x.dim() - 1)
        # print(x.size())
        if x.size(0) == 1:
            # These two lines run much faster in pytorch 0.4 than the two lines listed below.
            mean = x.view(-1).mean().view(*shape)
            std = x.view(-1).std().view(*shape)
        else:
            mean = x.view(x.size(0), -1).mean(1).view(*shape)
            std = x.view(x.size(0), -1).std(1).view(*shape)

        x = (x - mean) / (std + self.eps)

        if self.affine:
            shape = [1, -1] + [1] * (x.dim() - 2)
            x = x * self.gamma.view(*shape) + self.beta.view(*shape)
        return x


class MultiChannelActivation(nn.Module):
    def __init__(self, activation, output_dim):
        super(MultiChannelActivation, self).__init__()
        if isinstance(activation, str):
            activation = [activation]
        elif (not isinstance(activation, list)) or len(activation)!=output_dim:
            raise ValueError(f"Not valid activation {activation} for channels {output_dim}")

        self.activation = [get_activation(a) for a in activation]
    
    def forward(self, x):
        if len(self.activation)==1:
            if self.activation[0] is not None:
                return self.activation[0](x)
            else:
                return x
        else:
            x_list = [self.activation[i](x[:, i:i+1]) if self.activation[i] is not None else x[:, i:i+1] for i in range(x.shape[1])]
            x = torch.cat(x_list, dim=1)
            return x
            

def get_activation(activation):
        # initialize activation
    if activation == 'relu':
        return nn.ReLU(inplace=False)
    elif activation == '-relu':
        return NegReLU()
    elif activation == 'lrelu':
        return nn.LeakyReLU(0.2, inplace=True)
    elif activation == 'prelu':
        return nn.PReLU()
    elif activation == 'selu':
        return nn.SELU(inplace=True)
    elif activation == 'tanh':
        return nn.Tanh()
    elif activation == 'none':
        return None
    else:
        assert 0, "Unsupported activation: {}".format(activation)
        
class NegReLU(nn.Module):
    
    def __init__(self, inplace: bool = False):
        super(NegReLU, self).__init__()
        
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return - F.relu(input)
        
        
class LatLonPad(nn.Module):
    __constants__ = ['padding', 'lon_dim', 'lat_mode', 'value']

    def __init__(self, padding: _size_4_t, lon_dim: int, lat_mode: str = 'constant', value: float = 0.) -> None:
        super(LatLonPad, self).__init__()
        assert lat_mode in ['constant', 'circular']
        self.padding = _quadruple(padding)
        self.lon_dim = lon_dim
        self.lat_mode = lat_mode
        self.value = value

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        is1 = self.lon_dim==2 or self.lon_dim==-2
        is2 = self.lon_dim==3 or self.lon_dim==-1
        is_circular_pad = (is1, is1, is2, is2)
        circ_padding = tuple(p if b else 0 for p, b in zip(self.padding, is_circular_pad))
        zero_padding = tuple(p if not b else 0 for p, b in zip(self.padding, is_circular_pad))
        x = F.pad(input, circ_padding, mode='circular')
        x = F.pad(x, zero_padding, self.lat_mode, self.value)
        return x