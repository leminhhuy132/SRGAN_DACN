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
"""File description: Initialize the SRResNet model."""
import os
import shutil
import time
from enum import Enum

import numpy as np
import torch
from torch import nn
from torch import optim
from torch.cuda import amp
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import config
import imgproc
from dataset import CUDAPrefetcher, TrainValidImageDataset, TestImageDataset
from model import Generator
import matplotlib.pyplot as plt
from pytorch_ssim import ssim


def main():
    # Initialize training to generate network evaluation indicators
    best_psnr = 0.0

    train_prefetcher, valid_prefetcher, test_prefetcher = load_dataset()
    print("Load train dataset and valid dataset successfully.")

    model = build_model()
    print("Build SRResNet model successfully.")

    psnr_criterion, pixel_criterion = define_loss()
    print("Define all loss functions successfully.")

    optimizer = define_optimizer(model)
    print("Define all optimizer functions successfully.")

    print("Check whether the pretrained model is restored...")
    if config.resume:
        # Load checkpoint model
        checkpoint = torch.load(config.resume, map_location=lambda storage, loc: storage)
        # Restore the parameters in the training node to this point
        config.start_epoch = checkpoint["epoch"]
        best_psnr = checkpoint["best_psnr"]
        # Load checkpoint state dict. Extract the fitted model weights
        model_state_dict = model.state_dict()
        new_state_dict = {k: v for k, v in checkpoint["state_dict"].items() if k in model_state_dict}
        # Overwrite the pretrained model weights to the current model
        model_state_dict.update(new_state_dict)
        model.load_state_dict(model_state_dict)
        # Load the optimizer model
        optimizer.load_state_dict(checkpoint["optimizer"])
        # Load the scheduler model
        # scheduler.load_state_dict(checkpoint["scheduler"])
        print("Loaded pretrained model weights.")

    # Create a folder of super-resolution experiment results
    samples_dir = os.path.join("samples", config.exp_name)
    results_dir = os.path.join("results", config.exp_name)
    if not os.path.exists(samples_dir):
        os.makedirs(samples_dir)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # Create training process log file
    writer = SummaryWriter(os.path.join("samples", "logs", config.exp_name))

    # Initialize the gradient scaler
    scaler = amp.GradScaler()
    his_psnr = []
    his_ssim = []
    his_pixel_loss = []
    for epoch in range(config.start_epoch, config.epochs):
        train_loss = train(model, train_prefetcher, psnr_criterion, pixel_criterion, optimizer, epoch, scaler, writer)
        valid_psnr, valid_ssim = validate(model, valid_prefetcher, psnr_criterion, epoch, writer, "Valid")
        test_psnr, test_ssim = validate(model, test_prefetcher, psnr_criterion, epoch, writer, "Test")
        train_psnr, train_ssim, pixel_loss = train_loss
        his_psnr.append([train_psnr, valid_psnr, test_psnr])
        his_ssim.append([train_ssim, valid_ssim, test_ssim])
        his_pixel_loss.append(pixel_loss)
        print("\n")

        # Automatically save the model with the highest index
        is_best = test_psnr > best_psnr
        best_psnr = max(test_psnr, best_psnr)
        
        if is_best:
            torch.save({"epoch": epoch + 1,
                        "best_psnr": best_psnr,
                        "state_dict": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": None},
                        os.path.join(samples_dir, f"g_epoch_{epoch + 1}.pth.tar"))
            shutil.copyfile(os.path.join(samples_dir, f"g_epoch_{epoch + 1}.pth.tar"),
                            os.path.join(results_dir, "g_best.pth.tar"))
            # shutil.rmtree(samples_dir)
            # os.makedirs(samples_dir)
        if (epoch + 1) == config.epochs:
            torch.save({"epoch": epoch + 1,
                        "best_psnr": best_psnr,
                        "state_dict": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": None},
                       os.path.join(samples_dir, f"g_last.pth.tar"))
            shutil.copyfile(os.path.join(samples_dir, f"g_epoch_{epoch + 1}.pth.tar"),
                            os.path.join(results_dir, "g_last.pth.tar"))
        # plot
        plot(his_psnr, his_ssim, his_pixel_loss, samples_dir)


def load_dataset() -> [CUDAPrefetcher, CUDAPrefetcher, CUDAPrefetcher]:
    # Load train, test and valid datasets
    train_datasets = TrainValidImageDataset(config.train_image_dir, config.image_size, config.upscale_factor, "Train")
    valid_datasets = TrainValidImageDataset(config.valid_image_dir, config.image_size, config.upscale_factor, "Valid")
    test_datasets = TestImageDataset(config.test_lr_image_dir, config.test_hr_image_dir)

    # Generator all dataloader
    train_dataloader = DataLoader(train_datasets,
                                  batch_size=config.batch_size,
                                  shuffle=True,
                                  num_workers=config.num_workers,  # so tien trinh chay song song
                                  pin_memory=True,  # If you load your samples in the Dataset on CPU and would like to
                                                    # push it
                                                    # during training to the GPU, you can speed up the host to device
                                                    # transfer by enabling pin_memory.
                                  drop_last=True,  # parameter ignores the last batch (when the number of examples in
                                                   # your dataset is not divisible by your batch_size)
                                  persistent_workers=True)  # True will improve performances when you call into the dataloader
                                                            # multiple times in a row (as creating the workers is expensive).
    valid_dataloader = DataLoader(valid_datasets,
                                  batch_size=config.batch_size,
                                  shuffle=False,
                                  num_workers=config.num_workers,
                                  pin_memory=True,
                                  drop_last=False,
                                  persistent_workers=True)
    test_dataloader = DataLoader(test_datasets,
                                 batch_size=config.batch_size,
                                 shuffle=False,
                                 num_workers=config.num_workers,
                                 pin_memory=True,
                                 drop_last=False,
                                 persistent_workers=True)

    # Place all data on the preprocessing data loader
    train_prefetcher = CUDAPrefetcher(train_dataloader, config.device)
    valid_prefetcher = CUDAPrefetcher(valid_dataloader, config.device)
    test_prefetcher = CUDAPrefetcher(test_dataloader, config.device)

    return train_prefetcher, valid_prefetcher, test_prefetcher


def build_model() -> nn.Module:
    model = Generator().to(device=config.device, memory_format=torch.channels_last)
    return model


def define_loss() -> [nn.MSELoss, nn.MSELoss]:
    psnr_criterion = nn.MSELoss().to(device=config.device)
    pixel_criterion = nn.MSELoss().to(device=config.device)
    return psnr_criterion, pixel_criterion


def define_optimizer(model) -> optim.Adam:
    optimizer = optim.Adam(model.parameters(), config.model_lr, config.model_betas)
    return optimizer


def train(model,
          train_prefetcher,
          psnr_criterion,
          pixel_criterion,
          optimizer,
          epoch,
          scaler,
          writer) -> [float, float]:
    # Calculate how many iterations there are under epoch
    batches = len(train_prefetcher)

    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":6.6f")
    psnres = AverageMeter("PSNR", ":4.2f")
    ssimes = AverageMeter("SSIM", ":4.2f")
    progress = ProgressMeter(batches, [batch_time, data_time, losses, psnres, ssimes], prefix=f"Epoch: [{epoch + 1}]")

    # Put the generator in training mode
    model.train()

    batch_index = 0

    # Calculate the time it takes to test a batch of data
    end = time.time()
    # enable preload
    train_prefetcher.reset()
    batch_data = train_prefetcher.next()
    while batch_data is not None:
        # measure data loading time
        data_time.update(time.time() - end)

        lr = batch_data["lr"].to(device=config.device, memory_format=torch.channels_last, non_blocking=True)
        hr = batch_data["hr"].to(device=config.device, memory_format=torch.channels_last, non_blocking=True)

        # Initialize the generator gradient
        model.zero_grad(set_to_none=True)

        # Mixed precision training
        with amp.autocast():
            sr = model(lr)
            loss = pixel_criterion(sr, hr)

        # Gradient zoom
        scaler.scale(loss).backward()
        # Update generator weight
        scaler.step(optimizer)
        scaler.update()

        # measure accuracy and record loss
        psnr = 10. * torch.log10_(1. / psnr_criterion(sr, hr))
        losses.update(loss.item(), lr.size(0))
        psnres.update(psnr.item(), lr.size(0))

        ssim_score = ssim(sr.type_as(hr), hr)
        ssimes.update(ssim_score.item(), lr.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # Record training log information
        if batch_index % config.print_frequency == 0:
            # Writer Loss to file
            writer.add_scalar("Train/Loss", loss.item(), batch_index + epoch * batches + 1)
            progress.display(batch_index)

        # Preload the next batch of data
        batch_data = train_prefetcher.next()

        # After a batch of data is calculated, add 1 to the number of batches
        batch_index += 1
    loss_package = [psnres.avg, ssimes.avg, losses.avg]
    return loss_package


def validate(model, data_prefetcher, psnr_criterion, epoch, writer, mode) -> [float, float]:
    batch_time = AverageMeter("Time", ":6.3f")
    psnres = AverageMeter("PSNR", ":4.2f")
    ssimes = AverageMeter("SSIM", ":4.2f")
    progress = ProgressMeter(len(data_prefetcher), [batch_time, psnres, ssimes], prefix="Valid: ")

    # Put the model in verification mode
    model.eval()

    batch_index = 0

    # Calculate the time it takes to test a batch of data
    end = time.time()
    with torch.no_grad():
        # enable preload
        data_prefetcher.reset()
        batch_data = data_prefetcher.next()

        while batch_data is not None:
            # measure data loading time
            lr = batch_data["lr"].to(device=config.device, memory_format=torch.channels_last, non_blocking=True)
            hr = batch_data["hr"].to(device=config.device, memory_format=torch.channels_last, non_blocking=True)

            # Mixed precision
            with amp.autocast():
                sr = model(lr)

            # Convert RGB tensor to RGB image
            sr_image = imgproc.tensor2image(sr, range_norm=False, half=False)
            hr_image = imgproc.tensor2image(hr, range_norm=False, half=False)

            # Data range 0~255 to 0~1
            sr_image = sr_image.astype(np.float32) / 255.
            hr_image = hr_image.astype(np.float32) / 255.

            # RGB convert Y
            sr_y_image = imgproc.rgb2ycbcr(sr_image, use_y_channel=True)
            hr_y_image = imgproc.rgb2ycbcr(hr_image, use_y_channel=True)

            # Convert Y image to Y tensor
            sr_y_tensor = imgproc.image2tensor(sr_y_image, range_norm=False, half=False).unsqueeze_(0)
            hr_y_tensor = imgproc.image2tensor(hr_y_image, range_norm=False, half=False).unsqueeze_(0)

            # Convert CPU tensor to CUDA tensor
            sr_y_tensor = sr_y_tensor.to(device=config.device, memory_format=torch.channels_last, non_blocking=True)
            hr_y_tensor = hr_y_tensor.to(device=config.device, memory_format=torch.channels_last, non_blocking=True)

            # measure accuracy and record loss
            psnr = 10. * torch.log10_(1. / psnr_criterion(sr_y_tensor, hr_y_tensor))
            psnres.update(psnr.item(), lr.size(0))

            ssim_score = ssim(sr_y_tensor, hr_y_tensor)
            ssimes.update(ssim_score.item(), lr.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # Record training log information
            if batch_index % config.print_frequency == 0:
                progress.display(batch_index)

            # Preload the next batch of data
            batch_data = data_prefetcher.next()

            # After a batch of data is calculated, add 1 to the number of batches
            batch_index += 1

    # Print average PSNR metrics
    progress.display_summary()

    if mode == "Valid" or mode == "Test":
        writer.add_scalar(f"{mode}/PSNR", psnres.avg, epoch + 1)
    else:
        raise ValueError("Unsupported mode, please use `Valid` or `Test`.")

    return psnres.avg, ssimes.avg


def plot(his_psnr, his_ssim, his_pixel_loss, pathsave):
    psnr = np.array(his_psnr)
    plt.figure(1)
    plt.plot(psnr[:, 0], 'r')
    plt.plot(psnr[:, 1], 'y')
    plt.plot(psnr[:, 2], 'g')
    plt.legend(['train_psnr', 'valid_psnr', 'test_psnr'])
    plt.xlabel('Iter')
    plt.ylabel('PSNR score')
    plt.savefig(os.path.join(pathsave, 'psnr.png'))

    ssim = np.array(his_ssim)
    plt.figure(2)
    plt.plot(ssim[:, 0], 'r')
    plt.plot(ssim[:, 1], 'y')
    plt.plot(ssim[:, 2], 'g')
    plt.legend(['train_ssim', 'valid_ssim', 'test_ssim'])
    plt.xlabel('Iter')
    plt.ylabel('SSIM score')
    plt.savefig(os.path.join(pathsave, 'ssim.png'))

    plt.figure(3)
    plt.plot(his_pixel_loss, 'r')
    plt.legend(['Pixel Loss'])
    plt.xlabel('Iter')
    plt.ylabel('Pixel Loss')
    plt.savefig(os.path.join(pathsave, 'pixel_loss.png'))

# Copy form "https://github.com/pytorch/examples/blob/master/imagenet/main.py"
class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f", summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)

    def summary(self):
        if self.summary_type is Summary.NONE:
            fmtstr = ""
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = "{name} {avg:.2f}"
        elif self.summary_type is Summary.SUM:
            fmtstr = "{name} {sum:.2f}"
        elif self.summary_type is Summary.COUNT:
            fmtstr = "{name} {count:.2f}"
        else:
            raise ValueError(f"Invalid summary type {self.summary_type}")

        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print("\t".join(entries))

    def display_summary(self):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        print(" ".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"


if __name__ == "__main__":
    main()
