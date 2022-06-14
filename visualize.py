import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd


def plot3Resnet(his_psnr, his_ssim, his_pixel_loss, pathsave):
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


def plot3SRGAN(his_psnr, his_ssim, his_d_loss, his_content_loss, his_adversarial_loss, pathsave):
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

    d_loss = np.array(his_d_loss)
    plt.figure(3)
    plt.plot(d_loss[:, 0], 'b')
    plt.plot(d_loss[:, 1], 'g')
    plt.legend(['train_d_hr_loss', 'train_d_sr_loss'])
    plt.xlabel('Iter')
    plt.ylabel('D Loss')
    plt.savefig(os.path.join(pathsave, 'd_loss.png'))

    plt.figure(4)
    plt.plot(his_content_loss, 'r')
    plt.legend(['Content Loss'])
    plt.xlabel('Iters')
    plt.ylabel('Content Loss')
    plt.savefig(os.path.join(pathsave, 'content_loss.png'))

    plt.figure(5)
    plt.plot(his_adversarial_loss, 'r')
    plt.legend(['Adversarial Loss'])
    plt.xlabel('Iters')
    plt.ylabel('Adversarial Loss')
    plt.savefig(os.path.join(pathsave, 'adversarial_loss.png'))


def saveHisResnet(his_psnr, his_ssim, his_pixel_loss, pathData, pathSave):
    col = ['train_psnr', 'valid_psnr', 'test_psnr', 'train_ssim', 'valid_ssim', 'test_ssim', 'pixel_loss']
    if os.path.exists(os.path.join(pathData, 'hisResnetData.csv')):
        df = pd.read_csv(os.path.join(pathData, 'hisResnetData.csv'), index_col=0)
        data = np.concatenate((his_psnr, his_ssim, his_pixel_loss), axis=None)
        data = np.array([data])
        data = pd.DataFrame(data, columns=col)
        df = pd.concat([df, data], ignore_index=True, axis=0)
        df.to_csv(os.path.join(pathSave, 'hisResnetData.csv'))
    else:
        data = np.concatenate((his_psnr, his_ssim, his_pixel_loss), axis=1)
        df = pd.DataFrame(data, columns=col)
        df.to_csv(os.path.join(pathSave, 'hisResnetData.csv'))


def saveHisSRGAN(his_psnr, his_ssim, his_d_loss, his_content_loss, his_adversarial_loss, pathData, pathSave):
    col = ['train_psnr', 'valid_psnr', 'test_psnr', 'train_ssim', 'valid_ssim', 'test_ssim', 'd_hr_loss', 'd_sr_loss', 'content_loss', 'adversarial_loss']
    if os.path.exists(os.path.join(pathData, 'hisSRGANData.csv')):
        df = pd.read_csv(os.path.join(pathData, 'hisSRGANData.csv'), index_col=0)
        data = np.concatenate(his_psnr, his_ssim, his_d_loss, his_content_loss, his_adversarial_loss, axis=None)
        data = np.array([data])
        data = pd.DataFrame(data, columns=col)
        df = pd.concat([df, data], ignore_index=True, axis=0)
        df.to_csv(os.path.join(pathSave, 'hisSRGANData.csv'))
    else:
        data = np.concatenate(his_psnr, his_ssim, his_d_loss, his_content_loss, his_adversarial_loss, axis=1)
        df = pd.DataFrame(data, columns=col)
        df.to_csv(os.path.join(pathSave, 'hisSRGANData.csv'))


def plotResnet(pathData, pathSave):
    df = pd.read_csv(pathData, index_col=0)

    his_psnr = df[df.columns[0:3]].to_numpy()
    his_ssim = df[df.columns[3:6]].to_numpy()
    his_pixel_loss = df[df.columns[6]].to_numpy()
    plot3Resnet(his_psnr, his_ssim, his_pixel_loss, pathSave)


def plotSRGAN(pathData, pathSave):
    df = pd.read_csv(pathData, index_col=0)

    his_psnr = df[df.columns[0:3]].to_numpy()
    his_ssim = df[df.columns[3:6]].to_numpy()
    his_d_loss = df[df.columns[6:8]].to_numpy()
    his_content_loss = df[df.columns[8]].to_numpy()
    his_adversarial_loss = df[df.columns[9]].to_numpy()
    plot3SRGAN(his_psnr, his_ssim, his_d_loss, his_content_loss, his_adversarial_loss, pathSave)
