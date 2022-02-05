from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from tensorboardX import SummaryWriter
from tqdm import tqdm
from pathlib import Path
from datetime import datetime

import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn

import concurrent.futures
import multiprocessing
import itertools

from models import resnet
from autospeech_utils import set_path, create_logger, save_checkpoint, count_parameters
from data_objects.DeepSpeakerDataset import DeepSpeakerDataset
from data_objects.VoxcelebTestset import VoxcelebTestset
from functions import train_from_scratch, validate_verification
#from loss import CrossEntropyLoss

def train(a, bitwidth, seed): # a unused
    lr_min=0.001
    learning_rate=0.01
    num_workers=0
    num_classes=1211
    batch_size=256
    begin_epoch=0
    end_epoch=301
    val_freq=10
    print_freq=200
    load_path=None
    sub_dir="dev"
    partial_n_frames=300

    # cudnn related setting
    cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.enabled = True

    # Set the random seed manually for reproducibility.
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    subdir = "bitwidth_" + str(bitwidth) + "_" + datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')
    data_dir = "/home/nanoproj/ravit/speaker_verification/datasets/VoxCeleb1/" # "/mnt/usb/data/ravit/datasets/VoxCeleb1"

    log_dir = os.path.join("../logs/autospeech", subdir)
    if not os.path.isdir(log_dir):  # Create the log directory if it doesn't exist
        os.makedirs(log_dir)
    model_dir = os.path.join("../models/autospeech", subdir)
    if not os.path.isdir(model_dir):  # Create the model directory if it doesn't exist
        os.makedirs(model_dir)

    # model and optimizer
    model = resnet.resnet18(num_classes=num_classes, bitwidth=bitwidth)
    model = model.cuda()
    optimizer = optim.Adam(
        model.net_parameters() if hasattr(model, 'net_parameters') else model.parameters(),
        lr=0.01,
    )
    
    dummy_input = torch.zeros((1, 1, 300, 257))
    
    dummy_input = dummy_input.cuda()
    torch.onnx.export(model, dummy_input, os.path.join(model_dir, subdir + "_resnet18_verification.onnx"), verbose=False)

    # Loss
    #criterion = CrossEntropyLoss(num_classes).cuda()
    criterion = torch.nn.CrossEntropyLoss().cuda()

    # resume && make log dir and logger
    #load_path = "../models/autospeech/20220128-215932"
    checkpoint_file = None
    if load_path and os.path.exists(load_path):
        checkpoint_file = os.path.join(load_path, 'checkpoint_70.pth')
        assert os.path.exists(checkpoint_file)
        checkpoint = torch.load(checkpoint_file)

        # load checkpoint
        begin_epoch = checkpoint['epoch']
        last_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        best_eer = checkpoint['best_eer']
        optimizer.load_state_dict(checkpoint['optimizer'])
        #log_dir = checkpoint['path_helper']['log_path']
        #model_dir = checkpoint['path_helper']['ckpt_path']
        subdir = "20220128-215932"
        log_dir = os.path.join("../logs/autospeech/", subdir)
        model_dir = os.path.join("../models/autospeech/", subdir)
    else:
        begin_epoch = 0
        best_eer = 1.0
        last_epoch = -1
    
    logger = create_logger(log_dir, subdir)
    if checkpoint_file is not None:
        logger.info("=> loaded checkpoint '{}'".format(checkpoint_file))
    
    logger.info("Number of parameters: {}".format(count_parameters(model)))

    # dataloader
    train_dataset = DeepSpeakerDataset(
        Path(data_dir),  sub_dir, partial_n_frames)
    test_dataset_verification = VoxcelebTestset(
        Path(data_dir), partial_n_frames
    )
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, 
        num_workers=num_workers, pin_memory=True, shuffle=True, drop_last=True,)
    test_loader_verification = torch.utils.data.DataLoader(dataset=test_dataset_verification, batch_size=1, 
        num_workers=num_workers, pin_memory=True, shuffle=False, drop_last=False,)

    # training setting
    writer_dict = {
        'writer': SummaryWriter(log_dir),
        'train_global_steps': begin_epoch * len(train_loader),
        'valid_global_steps': begin_epoch // val_freq,
    }

    # training loop
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, end_epoch, lr_min,
        last_epoch=last_epoch
    )

    for epoch in tqdm(range(begin_epoch, end_epoch), desc='train progress'):
        model.train()
        train_from_scratch(model, optimizer, train_loader, criterion, epoch, writer_dict, learning_rate, print_freq, lr_scheduler)
        if epoch % val_freq == 0:
            eer = validate_verification(model, test_loader_verification)

            # remember best acc@1 and save checkpoint
            is_best = eer < best_eer
            best_eer = min(eer, best_eer)

            # save
            logger.info('=> saving checkpoint to {}'.format(model_dir))
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_eer': best_eer,
                'optimizer': optimizer.state_dict()
            }, is_best, model_dir, 'checkpoint_{}.pth'.format(epoch))
        lr_scheduler.step(epoch)

def main():
    bitwidth_options = [2, 3]
    sparsity_options = [0]
    model_combinations = list(itertools.product(bitwidth_options, sparsity_options))
    
    """
    with concurrent.futures.ProcessPoolExecutor() as executor:
        processes = []
        for (bitwidth, sparsity) in model_combinations:
            processes.append[executor.submit(train, bitwidth)]
    """
    """
    processes = []
    for (bitwidth, sparsity) in model_combinations:
        p = multiprocessing.Process(target=train, args=[bitwidth])
        p.start()
        processes.append(p)
    for process in processes:
        process.join()
    """
    for (bitwidth, sparsity) in model_combinations:
        torch.multiprocessing.spawn(train, args=(bitwidth,sparsity))

if __name__ == '__main__':
    main()