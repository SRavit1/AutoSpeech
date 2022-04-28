from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from sklearn.metrics import roc_curve

#from models.model import Network
from models import resnet, resnet_full, resnet_dense_xnor, resnet_quantized
#from config import cfg, update_config
#from autospeech_utils import create_logger#, Genotype
from data_objects.VoxcelebTestset import VoxcelebTestset
from functions import validate_verification, get_distances_labels_verification
from autospeech_utils import compute_eer

from matplotlib import pyplot as plt
import json

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
#data_dir = "/home/nanoproj/ravit/speaker_verification/datasets/VoxCeleb1/"
data_dir = "/mnt/usb/data/ravit/datasets/VoxCeleb1"

model_name = "full_20220420-060518"

checkpoint_name = "checkpoint_best.pth"
log_dir = os.path.join("../logs/autospeech", model_name)
load_path = os.path.join("../models/autospeech", model_name, checkpoint_name)
partial_n_frames = 300

# cudnn related setting
cudnn.benchmark = cudnn_benchmark
torch.backends.cudnn.deterministic = cudnn_deterministic
torch.backends.cudnn.enabled = cudnn_enabled

# Set the random seed manually for reproducibility.
#np.random.seed(seed)
#torch.manual_seed(seed)
#torch.cuda.manual_seed_all(seed)

# model and optimizer
binarized = False
quantized = False
bitwidth = 2
weight_bitwidth = 2
if not binarized and not quantized:
  model = resnet_full.resnet18(num_classes=num_classes, input_channels=1, normalize_output=False)
elif binarized:
  model = resnet_dense_xnor.resnet18(num_classes=num_classes, bitwidth=bitwidth, weight_bitwidth=weight_bitwidth, input_channels=1, normalize_output=False)
elif quantized:
  model = resnet_quantized.resnet18(num_classes=num_classes, bitwidth=bitwidth, weight_bitwidth=weight_bitwidth, input_channels=1, normalize_output=False)

model.eval()
#model.forward(torch.rand((1,1,300,257)))
model = model.cuda()

"""
#=====
binarized = False
quantized = True
bitwidth = 1
weight_bitwidth = 1
if not binarized and not quantized:
  model2 = resnet_full.resnet18(num_classes=num_classes, input_channels=1, normalize_output=False)
elif binarized:
  model2 = resnet_dense_xnor.resnet18(num_classes=num_classes, bitwidth=bitwidth, weight_bitwidth=weight_bitwidth, input_channels=1, normalize_output=False)
elif quantized:
  model2 = resnet_quantized.resnet18(num_classes=num_classes, bitwidth=bitwidth, weight_bitwidth=weight_bitwidth, input_channels=1, normalize_output=False)
print("model1 bitwidth", list(model.layer1.modules())[2].bitwidth)
print("model2 bitwidth before loaded checkpoint", list(model2.layer1.modules())[2].bitwidth)
model2.eval()
model2 = model2.cuda()
model2.load_state_dict(model.state_dict(), strict=False)
print("model2 bitwidth after loaded checkpoint", list(model2.layer1.modules())[2].bitwidth)
input_tensor = torch.ones((1, 1, 300, 257)).cuda()
with torch.no_grad():
    model_pred = model.forward(input_tensor)
    model2_pred = model2.forward(input_tensor)

np.testing.assert_array_equal(model_pred.cpu(), model2_pred.cpu(), err_msg="mismatched values")

print("model predictions equal")
#print("model prediction", model_pred.flatten()[:10])
#print("model2 prediction", model2_pred.flatten()[:10])

exit(0)
#=====
"""

# resume && make log dir and logger
if load_path and os.path.exists(load_path):
    checkpoint = torch.load(load_path)#, map_location=torch.device('cpu'))

    print("Epochs Trained", checkpoint["epoch"])
    print("Best EER Achieved", checkpoint["best_eer"])

    # load checkpoint
    load_result = model.load_state_dict(checkpoint['state_dict'])#, strict=False)
    print("load_state_dict output", load_result)
    #checkpoint['state_dict']['fc1.weight'] = checkpoint['state_dict']['classifier.weight']
    #checkpoint['state_dict']['fc1.bias'] = checkpoint['state_dict']['classifier.bias']
    #path_helper = checkpoint['path_helper']

    #logger = create_logger(os.path.dirname(load_path))
    #logger.info("=> loaded checkpoint '{}'".format(load_path))
else:
    raise AssertionError('Please specify the model to evaluate')

model.eval()
#print("model weights", model.conv1.weight.flatten()[:10])
input_tensor = torch.ones((1, 1, 300, 257)).cuda()
with torch.no_grad():
    model_pred = model.forward(input_tensor)
#print("model prediction", model_pred.flatten()[:10])
exit(0)

# dataloader
test_dataset_verification = VoxcelebTestset(
    Path(data_dir), partial_n_frames
)
test_loader_verification = torch.utils.data.DataLoader(
    dataset=test_dataset_verification,
    batch_size=1,
    num_workers=num_workers,
    pin_memory=True,
    shuffle=False,
    drop_last=False,
)
distances_labels_file = os.path.join(log_dir, "distances_labels.json")

#print("Begin evaluating", model_name, checkpoint_name)

calculate = True
if calculate:
    #eer = validate_verification(model, test_loader_verification, cuda=True)
    distances, labels = get_distances_labels_verification(model, test_loader_verification, cuda=True)
    if np.isnan(distances).any():
        print("NaN encountered in distances!")

    distances_labels_dict = {"distances": distances.tolist(), "labels": labels.tolist()}
    with open(distances_labels_file, "w") as f:
        json.dump(distances_labels_dict, f)
else:
    with open(distances_labels_file, "r") as f:
        distances_labels_dict = json.load(f)
        distances = np.array(distances_labels_dict["distances"])
        labels = np.array(distances_labels_dict["labels"])

fprs, tprs, thresholds = roc_curve(labels, distances)
thresholds[0] = max(distances)
eer = fprs[np.nanargmin(np.absolute((1 - tprs) - fprs))]

plot = True
if plot:
    fig_roc, ax_roc = plt.subplots()
    ax_roc.plot(thresholds, fprs, label="FP Curve")
    ax_roc.plot(thresholds, 1-tprs, label="FN Curve")
    ax_roc.set_title("Evaluation ROC")
    ax_roc.set_xlabel("Distances")
    ax_roc.set_ylabel("Error Rate")

    plt.legend()
    fig_roc.savefig(os.path.join(log_dir, checkpoint_name.split(".")[0] + "_roc.png"))
    plt.close('all')

print("distances min", distances.min())
print("distances max", distances.max())

print("Evaluation EER", eer)