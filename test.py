import pandas as pd
import os
import numpy as np
from visualize import plotResnet, saveHisResnet, saveHisSRGAN

his_psnr = [[1,2,3]]
his_ssim = [[2,3,4]]
his_pixel_loss = [3]
his_d_loss = [[6,7]]
his_content_loss = [9]
his_adversarial_loss = [10]
saveHisResnet(his_psnr[-1], his_ssim[-1], his_pixel_loss[-1], 'results/')
saveHisSRGAN(his_psnr[-1], his_ssim[-1], his_d_loss[-1], his_content_loss[-1], his_adversarial_loss[-1], 'results/')