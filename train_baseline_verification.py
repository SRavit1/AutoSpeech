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
import torchsummary

import concurrent.futures
import multiprocessing
import itertools

from models import resnet
from autospeech_utils import set_path, create_logger, save_checkpoint, count_parameters
from data_objects.DeepSpeakerDataset import DeepSpeakerDataset
from data_objects.VoxcelebTestset import VoxcelebTestset
from functions import train_from_scratch, validate_verification
#from loss import CrossEntropyLoss

cuda = True

def train(a, binarized, quantized, bitwidth, weight_bitwidth, sparsity): # a unused
    seed=0
    lr_min=0.001
    learning_rate=0.01
    num_workers=0
    num_classes=1211
    batch_size=128
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
    
    if not binarized and not quantized:
      prefix = "full_"
      subdir = prefix + datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')
    else:
      if binarized:
        prefix = "binarized_"
      elif quantized:
        prefix = "quantized_"
        
      subdir = prefix + "bitwidth_" + str(bitwidth) + "_weight_bitwidth_" + str(weight_bitwidth) + "_sparsity_" + str(sparsity) + "_" + datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')

    print("Model/Log Subdir", subdir)
    data_dir = "/mnt/usb/data/ravit/datasets/VoxCeleb1" #"/home/nanoproj/ravit/speaker_verification/datasets/VoxCeleb1/"

    log_dir = os.path.join("../logs/autospeech", subdir)
    if not os.path.isdir(log_dir):  # Create the log directory if it doesn't exist
        os.makedirs(log_dir)
    model_dir = os.path.join("../models/autospeech", subdir)
    if not os.path.isdir(model_dir):  # Create the model directory if it doesn't exist
        os.makedirs(model_dir)

    # model and optimizer
    model = resnet.resnet18(binarized=binarized, quantized=quantized, num_classes=num_classes, bitwidth=bitwidth, weight_bitwidth=weight_bitwidth, input_channels=1, normalize_output=False)
    #torchsummary.summary(model.cuda(), (1, 300, 257))
    if cuda:
        model = model.cuda()
    optimizer = optim.Adam(
        model.net_parameters() if hasattr(model, 'net_parameters') else model.parameters(),
        lr=0.01,
    )
    model.load_state_dict(torch.load("../models/autospeech/pretrained/checkpoint_best.pth"), strict=False)
    for p in model.modules():
      if hasattr(p, 'weight_org'):
        p.weight_org.copy_(p.weight.data)
    
    dummy_input = torch.zeros((1, 1, 300, 257))
    
    if cuda:
        dummy_input = dummy_input.cuda()
    torch.onnx.export(model, dummy_input, os.path.join(model_dir, subdir + "_resnet18_verification.onnx"), verbose=False)

    # Loss
    #criterion = CrossEntropyLoss(num_classes).cuda()
    criterion = torch.nn.CrossEntropyLoss()
    if cuda:
        criterion = criterion.cuda()

    # resume && make log dir and logger
    #subdir = "20220128-215932"
    #load_path = os.path.join("../models/autospeech", subdir)
    checkpoint_file = None
    if load_path and os.path.exists(load_path):
        checkpoint_file = os.path.join(load_path, 'checkpoint_60.pth')
        assert os.path.exists(checkpoint_file)
        checkpoint = torch.load(checkpoint_file)

        # load checkpoint
        begin_epoch = checkpoint['epoch']
        last_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        best_eer = checkpoint['best_eer']
        optimizer.load_state_dict(checkpoint['optimizer'])
        #log_dir = checkpoint['path_helper']['log_path']
        #model_dir = checkpoint['path_helper']['ckpt_path']
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
        num_workers=num_workers, pin_memory=cuda, shuffle=True, drop_last=True)
    test_loader_verification = torch.utils.data.DataLoader(dataset=test_dataset_verification, batch_size=1, 
        num_workers=num_workers, pin_memory=cuda, shuffle=False, drop_last=False)

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
        train_from_scratch(model, optimizer, train_loader, criterion, epoch, writer_dict, learning_rate, print_freq, lr_scheduler, cuda=cuda)
        if epoch % val_freq == 0:
            eer = validate_verification(model, test_loader_verification, cuda=cuda)

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
    """
    binarized_options = [False]
    quantized_options = [True]
    bitwidth_options = [32] #[1, 2, 3]
    sparsity_options = [0] #[0, 0.1]
    model_combinations = list(itertools.product(bitwidth_options, sparsity_options))
    """
    model_combinations = [(True, False, 4, 4, 0)] #binarized, quantized, activation bitwidth, weight bitwidth, sparsity

    for (binarized, quantized, bitwidth, weight_bitwidth, sparsity) in model_combinations:
        torch.multiprocessing.spawn(train, args=(binarized, quantized, bitwidth, weight_bitwidth, sparsity))

if __name__ == '__main__':
    main()
