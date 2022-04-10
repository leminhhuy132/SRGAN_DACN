# Copyright 2021 Dakewe Biotech Corporation. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import random

import numpy as np
import torch
from torch.backends import cudnn

# Random seed to maintain reproducible results
random.seed(0)
torch.manual_seed(0)
np.random.seed(0)
# Use GPU for training by default
device = torch.device("cuda", 0)
# Turning on when the image size does not change during training can speed up training
cudnn.benchmark = True
# Image magnification factor
upscale_factor = 4
# Current configuration parameter method
mode = "train_srresnet"
# Experiment name, easy to save weights and log files
exp_name = "SRResNet_baseline"

if mode == "train_srresnet":
    # Dataset address
    train_image_dir = "data/Textdata/train"
    valid_image_dir = "data/Textdata/train"
    test_lr_image_dir = f"data/Textdata/testx4"
    test_hr_image_dir = f"data/Textdata/test"


    image_size = 96
    batch_size = 16
    num_workers = 2

    # Incremental training and migration training
    start_epoch = 0
    resume = "results/SRGAN/srresnet-ImageNet-dbebcec6.pth.tar"

    # Total num epochs
    epochs = 45

    # Adam optimizer parameter
    model_lr = 1e-4
    model_betas = (0.9, 0.999)

    print_frequency = 1000

if mode == "train_srgan":
    # Dataset address
    train_image_dir = "data/Textdata/train"
    valid_image_dir = "data/Textdata/valid"
    test_lr_image_dir = f"data/Textdata/testx4"
    test_hr_image_dir = f"data/Textdata/test"

    image_size = 96
    batch_size = 16
    num_workers = 2

    # Incremental training and migration training
    start_epoch = 0
    resume = "results/SRGAN/srresnet-ImageNet-dbebcec6.pth.tar"
    resume_d = "results/SRGAN/discriminator-ImageNet-6113867a.pth.tar"
    resume_g = "results/SRGAN/srgan-ImageNet-839a8bfc.pth.tar"

    # Total num epochs
    epochs = 9

    # Loss function weight
    pixel_weight = 1.0
    content_weight = 0.006
    adversarial_weight = 0.001

    # Adam optimizer parameter
    model_lr = 1e-4
    model_betas = (0.9, 0.999)

    # MultiStepLR scheduler parameter
    optimizer_step_size = epochs // 2
    optimizer_gamma = 0.1

    print_frequency = 100

if mode == "valid":
    # Test data address
    lr_dir = f"data/Textdata/test"
    sr_dir = f"results/test/TEXT/{exp_name}"
    hr_dir = f"data/Textdata/test"

    model_path = f"results/SRGAN/srgan-ImageNet-839a8bfc.pth.tar"
