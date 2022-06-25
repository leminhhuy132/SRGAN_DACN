# Copyright 2022 Dakewe Biotech Corporation. All Rights Reserved.
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
import os

# Prepare dataset
os.system("python ./prepare_dataset.py --images_dir ../data/Textdata2/train --output_dir ../data/Textdata2/trainSRGAN --image_size 96 --step 48 --num_workers 1")
#
# Split train and valid
os.system("python ./split_train_valid_dataset.py --train_images_dir ../data/Textdata2/trainSRGAN --valid_images_dir ../data/Textdata2/validSRGAN --valid_samples_ratio 0.2")

os.system("python ./create_LRdata.py --images_dir ../data/Original --output_dir ../data/x4 --num_workers 1 --upscale_factor 4")
