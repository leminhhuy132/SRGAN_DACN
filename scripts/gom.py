import os
import shutil
import re
from glob import glob


list_folder_input = ['/home/minhhuy/Desktop/Python/SRGAN_DACN/data/realsr_textEdit/test',
                     '/home/minhhuy/Desktop/Python/SRGAN_DACN/data/datasets/EN/EN_test',
                     '/home/minhhuy/Desktop/Python/SRGAN_DACN/data/datasets/FR/FR_test',
                     '/home/minhhuy/Desktop/Python/SRGAN_DACN/data/datasets/VN/VN_test',
                     ]
output_dir = '/home/minhhuy/Desktop/Python/SRGAN_DACN/data/Textdata/test'

def main() -> None:
    # if os.path.exists(output_dir):
    #     shutil.rmtree(output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        start = 1
        print('Start: ', start)
    else:
        list_output = os.listdir(output_dir)
        print('Length Output: ', len(list_output))
        if len(list_output) > 0:
            list_output.sort(key=natural_keys)
            end_output = list_output[-1]
            end_output = end_output.split('.')[0]
            start = int(end_output) + 1
            print('Start: ', start)
        else:
            start = 1
            print('Start: ', start)

    index = start
    for folder_input in list_folder_input:
        # Get all image paths
        image_file_names = os.listdir(folder_input)
        print('Length Input {dir}: '.format(dir=folder_input), len(image_file_names))
        for image in image_file_names:
            worker(folder_input, image, index)
            index += 1


def worker(images_dir, image_file_name, index) -> None:
    new_image_name = f"{index}.{image_file_name.split('.')[-1]}"
    shutil.copyfile(f"{images_dir}/{image_file_name}", f"{output_dir}/{new_image_name}")

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [atoi(c) for c in re.split(r'(\d+)', text)]


if __name__ == "__main__":
    main()
