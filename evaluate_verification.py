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

from matplotlib import pyplot as plt

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

#load_path = "../models/autospeech/quantized_bitwidth_32_sparsity_0_20220224-123359/checkpoint_300.pth"
#load_path = "../models/autospeech/binarized_bitwidth_8_sparsity_0_20220306-095603/checkpoint_50.pth"
load_path = "../models/autospeech/binarized_bitwidth_1_sparsity_0_20220302-074431/checkpoint_60.pth"
#load_path = "../models/autospeech/pretrained/checkpoint_best.pth"
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
quantized=False
model = resnet.resnet18(num_classes=num_classes, binarized=False, quantized=quantized, input_channels=1, bitwidth=32)
model.forward(torch.rand((1,1,300,257)))
exit(0)
#model = model.cuda()

# resume && make log dir and logger
if load_path and os.path.exists(load_path):
    checkpoint = torch.load(load_path, map_location=torch.device('cpu'))

    # load checkpoint
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    checkpoint['state_dict']['fc1.weight'] = checkpoint['state_dict']['classifier.weight']
    checkpoint['state_dict']['fc1.bias'] = checkpoint['state_dict']['classifier.bias']
    #path_helper = checkpoint['path_helper']

    #logger = create_logger(os.path.dirname(load_path))
    #logger.info("=> loaded checkpoint '{}'".format(load_path))
else:
    raise AssertionError('Please specify the model to evaluate')

model_weights = {
  "conv1":model.conv1,
  "conv2_1": list(model.layer1.modules())[2],
  "conv2_2": list(model.layer1.modules())[5],
  "conv3_1": list(model.layer1.modules())[8],
  "conv3_2": list(model.layer1.modules())[11],
  "conv4_1": list(model.layer2.modules())[2],
  "conv4_2": list(model.layer2.modules())[5],
  "conv4_d": list(model.layer2.modules())[8],
  "conv5_1": list(model.layer2.modules())[11],
  "conv5_2": list(model.layer2.modules())[14],
  "conv6_1": list(model.layer2.modules())[2],
  "conv6_2": list(model.layer2.modules())[5],
  "conv6_d": list(model.layer2.modules())[8],
  "conv7_1": list(model.layer2.modules())[11],
  "conv7_2": list(model.layer2.modules())[14],
  "conv8_1": list(model.layer2.modules())[2],
  "conv8_2": list(model.layer2.modules())[5],
  "conv8_d": list(model.layer2.modules())[8],
  "conv9_1": list(model.layer2.modules())[11],
  "conv9_2": list(model.layer2.modules())[14],
  "fc1": model.fc
}

dists_dir = "./dists"
for (weight_name, weight_value) in model_weights.items():
  plt.clf()
  plt.hist(weight_value.weight.detach().flatten().numpy(), range=(-0.5, 0.5))
  plt.title(weight_name + " Weight Distribution")
  plt.xlabel("Weight value")
  plt.ylabel("Frequency")
  weight_dist_file = os.path.join(dists_dir, weight_name + "_dist.png")
  plt.savefig(weight_dist_file)

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

#validate_verification(model, test_loader_verification)
