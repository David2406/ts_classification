import numpy as np
import torch
from __init__ import volta_logger


def rocket_ks_dilations(ts_length = 5000,
                        nb_ks = 1000,
                        ks_bounds = None,
                        save_outputs = False,
                        kernel_sizes_file_path = './rocket_kernel_sizes.pt',
                        dilations_file_path = './rocket_dilations.pt',
                        logger = volta_logger):

    logger.info('Generating kernel sizes and dilations to build Rocket random convolution features')

    if ks_bounds is None:
        ks_min = 3
        ks_max = int(ts_length / 2)
    else:
        ks_min, ks_max = ks_bounds

    kernel_sizes = np.random.choice(np.arange(ks_min, ks_max + 1, dtype = int), nb_ks, replace = True)
    dilation_pow_bounds = np.log2((ts_length - 1) / (kernel_sizes - 1))
    dilation_pows = np.stack([np.random.uniform(0., float(a), 1) for a in dilation_pow_bounds])
    dilations = np.floor(2. ** (dilation_pows)).astype(int).flatten()

    if save_outputs:
        torch.save(kernel_sizes, kernel_sizes_file_path)
        torch.save(dilations, dilations_file_path)

    return kernel_sizes, dilations
