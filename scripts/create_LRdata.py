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
import argparse
import multiprocessing
import os
import shutil

import cv2
import numpy as np
from tqdm import tqdm


def main(args) -> None:
    if os.path.exists(args.output_dir):
        shutil.rmtree(args.output_dir)
    os.makedirs(args.output_dir)

    # Get all image paths
    image_file_names = os.listdir(args.images_dir)

    # Splitting images with multiple threads
    progress_bar = tqdm(total=len(image_file_names), unit="image", desc="Resize image")
    workers_pool = multiprocessing.Pool(args.num_workers)
    for image_file_name in image_file_names:
        workers_pool.apply_async(worker, args=(image_file_name, args), callback=lambda arg: progress_bar.update(1))
    workers_pool.close()
    workers_pool.join()
    progress_bar.close()


def worker(image_file_name, args) -> None:
    image = cv2.imread(f"{args.images_dir}/{image_file_name}", cv2.IMREAD_UNCHANGED)
    height, width = image.shape[0:2]
    newsameH, newsameW = (height//args.upscale_factor)*args.upscale_factor, (width//args.upscale_factor)*args.upscale_factor
    image_original = cv2.resize(image, (newsameW, newsameH), interpolation=cv2.INTER_CUBIC)
    image_scale = cv2.resize(image_original, (newsameW//args.upscale_factor, newsameH//args.upscale_factor),
                       interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(f"{args.images_dir}/{image_file_name}", image_original)
    cv2.imwrite(f"{args.output_dir}/{image_file_name}", image_scale)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare database scripts.")
    parser.add_argument("--images_dir", type=str, help="Path to input image directory.")
    parser.add_argument("--output_dir", type=str, help="Path to generator image directory.")
    parser.add_argument("--num_workers", type=int, help="How many threads to open at the same time.")
    parser.add_argument("--upscale_factor", type=int, help="Image zoom factor.")
    args = parser.parse_args()

    main(args)
