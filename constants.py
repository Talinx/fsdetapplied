import logging
import os
from datetime import datetime

import torch
import torch.optim as optim

batch_size = 16
eta = 0.000001
epochs = 1000
k = 1

num_classes = 23 + 2

batch_size_validation = 16
inference_batch_size = 32

display_every_batches = 5
logs_path = 'logs'
losses_path = 'losses'
models_path = 'models'
inferenced_img_path = 'labeled images'

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
cpu_device = torch.device('cpu')


console_logger_added = False


def get_logger(suffix: str=None):
    if suffix is None:
        suffix = str(datetime.now())
    logger = logging.getLogger('FsDet')
    os.makedirs(logs_path, exist_ok=True)
    log_path = os.path.join(logs_path, f'training {suffix}.log')
    logging.basicConfig(filename=log_path, level=logging.INFO, force=True)
    global console_logger_added
    if not console_logger_added:
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        logger.addHandler(console)
        console_logger_added = True
    return logger, log_path, suffix


def get_default_optimizer(parameters):
    """Get the default optimizer

    Returns
    -------
    default optimizer
    """
    return optim.Adam(parameters, lr=eta)
