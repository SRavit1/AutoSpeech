from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn

#from models.model import Network
from models import resnet
#from config import cfg, update_config
#from autospeech_utils import create_logger#, Genotype
from data_objects.VoxcelebTestset import VoxcelebTestset
from functions import validate_verification

def main():
    #args = parse_args()
    #update_config(cfg, args)
    #if load_path is None:
    #    raise AttributeError("Please specify load path.")

    seed = 0

    cudnn_benchmark = True
    cudnn_deterministic = False
    cudnn_enabled = False

    num_workers = 0
    num_classes = 1211
    data_dir = "/home/nanoproj/ravit/speaker_verification/datasets/VoxCeleb1/"

    #load_path = "../models/autospeech/20220128-215932/checkpoint_best.pth"
    load_path = "../models/autospeech/bitwidth_32_sparsity_0_20220214-223514/checkpoint_40.pth"
    #load_path = "../models/autospeech/bitwidth_8_sparsity_0_20220208-122531/checkpoint_300.pth"
    partial_n_frames = 300

    # cudnn related setting
    cudnn.benchmark = cudnn_benchmark
    torch.backends.cudnn.deterministic = cudnn_deterministic
    torch.backends.cudnn.enabled = cudnn_enabled

    # Set the random seed manually for reproducibility.
    np.random.seed(seed)
    torch.manual_seed(seed)
    #torch.cuda.manual_seed_all(seed)

    # model and optimizer
    model = resnet.resnet18(num_classes=num_classes)
    #model = model.cuda()

    # resume && make log dir and logger
    if load_path and os.path.exists(load_path):
        checkpoint = torch.load(load_path, map_location=torch.device('cpu'))

        # load checkpoint
        model.load_state_dict(checkpoint['state_dict'])
        #path_helper = checkpoint['path_helper']

        #logger = create_logger(os.path.dirname(load_path))
        #logger.info("=> loaded checkpoint '{}'".format(load_path))
    else:
        raise AssertionError('Please specify the model to evaluate')

    # dataloader
    test_dataset_verification = VoxcelebTestset(
        Path(data_dir), partial_n_frames
    )
    test_loader_verification = torch.utils.data.DataLoader(
        dataset=test_dataset_verification,
        batch_size=1,
        num_workers=num_workers,
        #pin_memory=True,
        shuffle=False,
        drop_last=False,
    )

    validate_verification(model, test_loader_verification)



if __name__ == '__main__':
    main()
