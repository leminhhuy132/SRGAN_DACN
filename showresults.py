import cv2
import os
from natsort import natsorted

original_dir = 'data/Original'
data_dir = ''
results_dir = 'results/test/SRRESNET'
results_dir2 = 'results/test/SRRESNET100'
save_dir = ''


def ResizeWithAspectRatio(image, widthWindown=None, heightWindown=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    # if width is None and height is None:
    #     return image
    # if width is None:
    #     r = height / float(h)
    #     dim = (int(w * r), height)
    # else:
    #     r = width / float(w)
    #     dim = (width, int(h * r))

    if widthWindown is None and heightWindown is None:
        return image
    if widthWindown is not None and heightWindown is not None:
        rh = float(h) / heightWindown
        rw = float(w) / widthWindown
        if rh > rw:
            dim = (int(w / round(rh, 1)), int(h / round(rh, 1)))
        else:
            dim = (int(w / round(rw, 1)), int(h / round(rw, 1)))
    return cv2.resize(image, dim, interpolation=inter)


list_results = natsorted(os.listdir(results_dir))
if data_dir:
    list_data = natsorted(os.listdir(data_dir))
    if len(list_data) != len(list_results):
        print('Data and Results are not the same length')
if original_dir:
    list_original = natsorted(os.listdir(original_dir))
    if len(original_dir) != len(list_results):
        print('Original and Results are not the same length')
if results_dir2:
    list_results2 = natsorted(os.listdir(results_dir2))
    if len(list_results2) != len(list_results):
        print('Results 2 and Results are not the same length')
if save_dir:
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
i = 0
if data_dir and original_dir:
    print('Fill data_dir or original_dir')
else:
    while i < len(list_results):
        if data_dir:
            img_data = cv2.imread(f'{data_dir}/{list_data[i]}')
            hd, wd, _ = img_data.shape
        if original_dir:
            img_original = cv2.imread(f'{original_dir}/{list_original[i]}')
        if results_dir2:
            img_result2 = cv2.imread(f'{results_dir2}/{list_results2[i]}')
        img_result = cv2.imread(f'{results_dir}/{list_results[i]}')
        hr, wr, _ = img_result.shape

        # top = (hr - hd)//2
        # bottom = (hr - hd)//2
        # left = (wr - wd)//2
        # right = (wr - wd)//2
        if data_dir and original_dir:
            top = (hr - hd)//2
            bottom = (hr - hd)//2
            left = 20
            right = 20
            if (hr - hd) % 2:
                top += 1
            if (wr - wd) % 2:
                left += 1

            constant = cv2.copyMakeBorder(img_data, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                          value=(255, 255, 255))
            if results_dir2:
                img = cv2.hconcat([img_original, constant, img_result, img_result2])
            else:
                img = cv2.hconcat([img_original, constant, img_result])
        elif data_dir:
            top = (hr - hd) // 2
            bottom = (hr - hd) // 2
            left = 20
            right = 20
            if (hr - hd) % 2:
                top += 1
            if (wr - wd) % 2:
                left += 1

            constant = cv2.copyMakeBorder(img_data, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                          value=(255, 255, 255))
            if results_dir2:
                img = cv2.hconcat([constant, img_result, img_result2])
            else:
                img = cv2.hconcat([constant, img_result])
        else:
            if results_dir2:
                img = cv2.hconcat([img_original, img_result, img_result2])
            else:
                img = cv2.hconcat([img_original, img_result])
        if save_dir:
            print(os.path.join(save_dir, list_results[i]))
            cv2.imwrite(os.path.join(save_dir, list_results[i]), img)

        img = ResizeWithAspectRatio(img, widthWindown=1850, heightWindown=1080)  # Resize by width OR
        # resize = ResizeWithAspectRatio(image, height=1280) # Resize by height
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


