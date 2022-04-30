import cv2
import os
from glob import glob
import sys
import matplotlib.pyplot as plt


data_dir = 'data/Textdata/testx4'
results_dir = 'results/test/Text/SRGAN'
list_data = os.listdir(data_dir)
list_results = os.listdir(results_dir)

i = 0
if len(list_data) != len(list_results):
    print('Data and Results are not the same length')
else:

    while i < len(list_data):
        img_data = cv2.imread(f'{data_dir}/{list_data[i]}')
        img_result = cv2.imread(f'{results_dir}/{list_results[i]}')
        hd, wd, _ = img_data.shape
        hr, wr, _ = img_result.shape
        top = (hr - hd)//2
        bottom = (hr - hd)//2
        left = (wr - wd)//2
        right = (wr - wd)//2
        if (hr - hd) % 2:
            top += 1
        if (wr - wd) % 2:
            left += 1

        constant = cv2.copyMakeBorder(img_data, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))

        img = cv2.hconcat([constant, img_result])
        cv2.imshow('Display', img)

        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('a'):
            i -= 1
            continue
        elif key == ord('d'):
            i += 1
            continue
