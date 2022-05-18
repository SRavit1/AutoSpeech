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
from matplotlib import pyplot as plt
import json

import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
#import torchsummary

import concurrent.futures
import multiprocessing
import itertools

from models import resnet, resnet_full, resnet_quantized, resnet_dense_xnor
from autospeech_utils import set_path, create_logger, save_checkpoint, count_parameters
from data_objects.DeepSpeakerDataset import DeepSpeakerDataset
from data_objects.VoxcelebTestset import VoxcelebTestset
from functions import train_from_scratch, validate_verification
#from loss import CrossEntropyLoss

cuda = True
seed=0
lr_min=1e-4
learning_rate=1e-3
num_workers=0
num_classes=1211
batch_size=128
begin_epoch=0
end_epoch=501
pretrain_epoch = 50
val_freq=10
print_freq=200
load_path=None
sub_dir="dev"
partial_n_frames=300

#def train(a, xnor_quantized, fp_quantized, abw_history, wbw_history, sparsity): # a unused
def train(xnor_quantized, fp_quantized, abw_history, wbw_history, sparsity):
    # cudnn related setting
    cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.enabled = True

    # Set the random seed manually for reproducibility.
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    if not xnor_quantized and not fp_quantized:
      prefix = "full"
      subdir = prefix + datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')
    else:
      if xnor_quantized:
        prefix = "xnor"
      elif fp_quantized:
        prefix = "fp"
        
      subdir = prefix
      subdir += "_abw_" + str(abw_history[-1])
      subdir += "_wbw_" + str(wbw_history[-1])
      if sparsity > 0:
        subdir += "_sparsity_" + str(sparsity)
      subdir += "_" + datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')

    print("Model/Log Subdir", subdir)
    with open("./dataset.txt", 'r') as f:
        data_dir = f.read().strip()

    log_dir = os.path.join("../logs/autospeech", subdir)
    if not os.path.isdir(log_dir):  # Create the log directory if it doesn't exist
        os.makedirs(log_dir)
    #model_dir = os.path.join("../models/autospeech", subdir)
    model_dir = os.path.join(log_dir, "models")
    if not os.path.isdir(model_dir):  # Create the model directory if it doesn't exist
        os.makedirs(model_dir)

    # model and optimizer
    if not xnor_quantized and not fp_quantized:
      model = resnet_full.resnet18(num_classes=num_classes, input_channels=1, normalize_output=False)
    elif xnor_quantized:
      model = resnet_dense_xnor.resnet18(num_classes=num_classes, bitwidth=abw_history[-1], weight_bitwidth=wbw_history[-1], input_channels=1, normalize_output=False)
    elif fp_quantized:
      model = resnet_quantized.resnet18(num_classes=num_classes, bitwidth=abw_history[-1], weight_bitwidth=wbw_history[-1], input_channels=1, normalize_output=False)

    #torchsummary.summary(model.cuda(), (1, 300, 257))
    if cuda:
        model = model.cuda()
    optimizer = optim.Adam(
        model.net_parameters() if hasattr(model, 'net_parameters') else model.parameters(),
        lr=learning_rate,
    )

    #Tianmu: Can simply change bitwidth of each layer (no need to copy)
    #initialize model weights using existing resnet model
    
    #pretrained_model_dir = "full_20220413-054624"
    #pretrained_ckpt = "checkpoint_best.pth"
    pretrained_model_dir = None
    pretrained_ckpt = None
    if pretrained_model_dir and pretrained_ckpt:
        pretrained_model_path = os.path.join(pretrained_model_dir, pretrained_ckpt)
        model.load_state_dict(torch.load(os.path.join("../models/autospeech/", pretrained_model_path))["state_dict"], strict=False)
    
    model_data = {
        "type": prefix,
        "pretrained_model_dir": pretrained_model_dir,
        "pretrained_ckpt": pretrained_ckpt,
        "abw_history": abw_history,
        "wbw_history": wbw_history,
        "sparsity": sparsity,
    }
    model_data_path = os.path.join(log_dir, "metadata.json")
    with open(model_data_path, 'w') as f:
        json.dump(model_data, f, indent=4)

    dummy_input = torch.zeros((1, 1, 300, 257))
    if cuda:
        dummy_input = dummy_input.cuda()
    torch.onnx.export(model, dummy_input, os.path.join(model_dir, subdir + "_resnet18_verification.onnx"), verbose=False)

    save_checkpoint({
        'epoch': 0,
        'state_dict': model.state_dict(),
        'best_eer': 0,
        'optimizer': optimizer.state_dict()
    }, False, model_dir, 'checkpoint_init.pth')
    print("Saved initial model path")

    # Loss
    #criterion = CrossEntropyLoss(num_classes).cuda()
    criterion = torch.nn.CrossEntropyLoss()
    if cuda:
        criterion = criterion.cuda()

    begin_epoch = 0
    best_eer = 1.0
    last_epoch = -1
    
    logger = create_logger(log_dir, subdir)
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
        #last_epoch=last_epoch
    )

    #TODO: Save raw data in JSON, in case we want to generate more nuanced graph
    loss_history = []
    top1_history = []
    top5_history = []
    eer_history = []
    if xnor_quantized or fp_quantized:
        epochs = list(range(1, end_epoch))
        fig_bw, ax_bw = plt.subplots()
        ax_bw.plot(epochs, abw_history, label='activation bw')
        ax_bw.plot(epochs, wbw_history, label='weight bw')
        ax_bw.set_title("Bitwidth Adjustment")
        ax_bw.set_xlabel("Epochs")
        ax_bw.set_ylabel("Bitwidth")
        ax_bw.legend()
        fig_bw.savefig(os.path.join(log_dir, "bw_adjustment.png"))

        plt.close('all')
    
    for epoch in tqdm(range(begin_epoch, end_epoch), desc='train progress'):
        if xnor_quantized or fp_quantized:
            if epoch==0 or (not abw_history[epoch]==abw_history[epoch-1]) or (not wbw_history[epoch]==wbw_history[epoch-1]):
                logger.info("Activation bw: {}, Weight bw: {}".format(abw_history[epoch], wbw_history[epoch]))
            model.change_bitwidth(abw_history[epoch], wbw_history[epoch])

        model.train()
        top1, top5, loss = train_from_scratch(model, optimizer, train_loader, criterion, epoch, writer_dict, learning_rate, print_freq, lr_scheduler, cuda=cuda)
        loss_history.append(float(loss))
        top1_history.append(float(top1))
        top5_history.append(float(top5))
        logger.info("Epoch average loss {}".format(loss))
        logger.info("Epoch average top1 {}".format(top1))
        logger.info("Epoch average top5 {}".format(top5))

        if (not epoch == 0) and epoch % val_freq == 0:
            model.eval()
            eer = validate_verification(model, test_loader_verification, cuda=cuda)
            eer_history.append(float(eer))

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

        epochs = list(range(1, epoch+2))
        epochs_eval = list(range(1+val_freq, epoch+2, val_freq))

        model_data["epochs"] = epochs
        model_data["epochs_eval"] = epochs_eval
        model_data["loss_history"] = loss_history
        model_data["top1_history"] = top1_history
        model_data["top5_history"] = top5_history
        model_data["eer_history"] = eer_history
        with open(model_data_path, 'w') as f:
            json.dump(model_data, f, indent=4)

        #Visualization
        fig_loss, ax_loss = plt.subplots()
        ax_loss.plot(epochs, loss_history)
        ax_loss.set_title("Training Loss Convergence")
        ax_loss.set_xlabel("Epochs")
        ax_loss.set_ylabel("Training Loss")
        fig_loss.savefig(os.path.join(log_dir, "loss_convergence.png"))
        fig_top1, ax_top1 = plt.subplots()
        ax_top1.plot(epochs, top1_history)
        ax_top1.set_title("Training Top1 Accuracy Convergence")
        ax_top1.set_xlabel("Epochs")
        ax_top1.set_ylabel("Training Top1")
        fig_top1.savefig(os.path.join(log_dir, "top1_convergence.png"))
        fig_top5, ax_top5 = plt.subplots()
        ax_top5.plot(epochs, top5_history)
        ax_top5.set_title("Training Top5 Accuracy Congergence")
        ax_top5.set_xlabel("Epochs")
        ax_top5.set_ylabel("Training Top5")
        fig_top5.savefig(os.path.join(log_dir, "top5_convergence.png"))
        fig_eer, ax_eer = plt.subplots()
        ax_eer.plot(epochs_eval, np.array(eer_history)*100)
        ax_eer.set_title("Evaluation EER Convergence")
        ax_eer.set_xlabel("Epochs")
        ax_eer.set_ylabel("EER")
        fig_eer.savefig(os.path.join(log_dir, "eer_convergence.png"))
        plt.close('all')

def parse_args():
    parser = argparse.ArgumentParser(description='Train binarized/fixed point/full precision AutoSpeech.')
    # general
    parser.add_argument('--xnor',
                        help='use xnor neural network',
                        action='store_true')
    parser.add_argument('--fp',
                        help='use fixed point (fp) neural network',
                        action='store_true')
    parser.add_argument('--abw_final',
                        help='activation bw',
                        default=2,
                        type=int)
    parser.add_argument('--wbw_final',
                        help='weight bw',
                        default=2,
                        type=int)
    parser.add_argument('--sparsity',
                        help='weight bw',
                        default=0,
                        type=float)

    args = parser.parse_args()

    return args

def main():
    args = parse_args()

    abw_history = [4]*25+[args.abw_final]*(end_epoch-begin_epoch-1-25)
    wbw_history = [4]*25+[args.wbw_final]*(end_epoch-begin_epoch-1-25)

    model_combinations = [(args.xnor, args.fp, abw_history, wbw_history, args.sparsity)] #xnor_quantized, fp_quantized, pretrain abw, pretrain wbw, sparsity

    for (xnor_quantized, fp_quantized, abw_history, wbw_history, sparsity) in model_combinations:
        train(xnor_quantized, fp_quantized, abw_history, wbw_history, sparsity)

if __name__ == '__main__':
    main()
