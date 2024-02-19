import torch
import torch.nn as nn
from __init__ import volta_logger


def build_rocket_1d_convs(kernel_sizes,
                          dilations,
                          in_channels = 12,
                          out_channels = 1,
                          padding = 'same',
                          stride = 1,
                          weight_init_bounds = [-1., 1.],
                          bias_init_params = [0., 1.],
                          ** kwargs):

    rocket_1d_convs = []
    for ks, d in zip(kernel_sizes, dilations):
        conv = torch.nn.Conv1d(in_channels,
                               out_channels,
                               ks,
                               stride,
                               padding,
                               d)
        torch.nn.init.uniform_(conv.weight,
                               a = weight_init_bounds[0],
                               b = weight_init_bounds[1])
        torch.nn.init.normal_(conv.bias,
                              mean = bias_init_params[0],
                              std = bias_init_params[1])

        conv.weight.requires_grad = False
        conv.bias.requires_grad = False

        rocket_1d_convs.append(conv)

    return rocket_1d_convs


class RocketFeatures(nn.Module):
    def __init__(self,
                 kernel_sizes,
                 dilations,
                 logger = volta_logger,
                 **kwargs):
        super().__init__()

        self.logger = logger
        self.nb_kernels = kernel_sizes.shape[0]

        self.ppv_op = lambda conv: torch.sum(conv > 0, dim = 2, keepdim = True) / conv.shape[2]
        self.max_op = lambda conv: torch.nn.MaxPool1d(conv.shape[2], stride = 1, padding = 0, dilation = 1)(conv)
        self.norm_op = torch.nn.BatchNorm1d(2 * self.nb_kernels, eps = 1e-05, momentum = 0.1, affine = False)

        self.rocket_1d_convs = nn.ModuleList(build_rocket_1d_convs(kernel_sizes,
                                                                   dilations,
                                                                   **kwargs))
        self.nb_feats = len(self.rocket_1d_convs)

    def build_features(self,
                       x,
                       print_freq = 10,
                       save_freq = None,
                       save_file_path = './rocket_feats.pkl',
                       logger = volta_logger):

        x = torch.tensor(x,
                         device = x.device,
                         dtype = torch.float32,
                         requires_grad = False)
        nb_obs = x.shape[0]

        rocket_features = []

        for i, rocket_1d_conv in enumerate(self.rocket_1d_convs):

            if i % print_freq == 0:
                self.logger.info('Building rocket features for kernel %d', i)

            conv_x = rocket_1d_conv(x)
            rocket_features.append(self.ppv_op(conv_x).reshape(nb_obs, 1))
            rocket_features.append(self.max_op(conv_x).reshape(nb_obs, 1))

            if not (save_freq is None) and (i % save_freq == 0):
                logger.info('Saving rocket features at kernel %d', i)
                loc_rocket_features = self.norm_op(torch.cat(rocket_features, dim = 1))
                torch.save(loc_rocket_features, save_file_path)

        rocket_features = torch.cat(rocket_features, dim = 1)

        if not (save_freq is None):
            torch.save(rocket_features, save_file_path)

        return self.norm_op(rocket_features)


class ConvBlock(torch.nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 padding,
                 bn_eps = 1e-3,
                 bn_mom = 0.99,
                 pool_layer_op = torch.nn.AvgPool1d(2),
                 activ_fun = torch.nn.ReLU(),
                 ** kargs):

        super(ConvBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.pool_op = pool_layer_op
        self.conv = torch.nn.Conv1d(in_channels,
                                    out_channels,
                                    kernel_size,
                                    padding = padding)
        self.bn = torch.nn.BatchNorm1d(out_channels,
                                       eps = bn_eps,
                                       momentum = bn_mom)
        self.activ_fun = activ_fun

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activ_fun(x)
        return self.pool_op(x)
