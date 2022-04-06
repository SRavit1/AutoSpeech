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

model_name = "quantized_bitwidth_1_weight_bitwidth_1_sparsity_0_20220404-212154"
#load_path = os.path.join("../models/autospeech", model_name, "checkpoint_best.pth")
load_path = os.path.join("../models/autospeech", model_name, "checkpoint_init.pth")
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
model = resnet.resnet18(num_classes=num_classes, binarized=False, quantized=True, input_channels=1, bitwidth=2)
model.eval()
#model = model.cuda()

#TODO: Replace above block with this code
# model and optimizer
"""
if not binarized and not quantized:
  model = resnet_full.resnet18(num_classes=num_classes, input_channels=1, normalize_output=False)
elif binarized:
  model = resnet_dense_xnor.resnet18(num_classes=num_classes, bitwidth=bitwidth, weight_bitwidth=weight_bitwidth, input_channels=1, normalize_output=False)
elif quantized:
  model = resnet_quantized.resnet18(num_classes=num_classes, bitwidth=bitwidth, weight_bitwidth=weight_bitwidth, input_channels=1, normalize_output=False)
"""

# resume && make log dir and logger
if load_path and os.path.exists(load_path):
    checkpoint = torch.load(load_path, map_location=torch.device('cpu'))
    # load checkpoint
    if 'classifier.weight' in checkpoint['state_dict'].keys():
      checkpoint['state_dict']['fc1.weight'] = checkpoint['state_dict']['classifier.weight']
    if 'classifier.bias' in checkpoint['state_dict'].keys():
      checkpoint['state_dict']['fc1.bias'] = checkpoint['state_dict']['classifier.bias']
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    #path_helper = checkpoint['path_helper']
    #logger = create_logger(os.path.dirname(load_path))
    #logger.info("=> loaded checkpoint '{}'".format(load_path))
else:
    raise AssertionError('Please specify the model to evaluate')

#model.forward(torch.rand((1,1,300,257)))
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
  "conv6_1": list(model.layer3.modules())[2],
  "conv6_2": list(model.layer3.modules())[5],
  "conv6_d": list(model.layer3.modules())[8],
  "conv7_1": list(model.layer3.modules())[11],
  "conv7_2": list(model.layer3.modules())[14],
  "conv8_1": list(model.layer4.modules())[2],
  "conv8_2": list(model.layer4.modules())[5],
  "conv8_d": list(model.layer4.modules())[8],
  "conv9_1": list(model.layer4.modules())[11],
  "conv9_2": list(model.layer4.modules())[14],
  "fc1": model.fc
}

dists_dir = os.path.abspath(os.path.join(load_path, os.pardir, "dists"))
print("Saving weight distributions to", dists_dir)
if not os.path.exists(dists_dir):
  os.makedirs(dists_dir)
for (weight_name, weight_value) in model_weights.items():
  plt.clf()
  plt.hist(weight_value.weight.detach().flatten().numpy())
  plt.title(weight_name + " Weight Distribution")
  plt.xlabel("Weight value")
  plt.ylabel("Frequency")
  weight_dist_file = os.path.join(dists_dir, weight_name + "_dist.png")
  plt.savefig(weight_dist_file)
