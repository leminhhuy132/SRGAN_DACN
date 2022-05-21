import os
import shutil
import re
from glob import glob
import argparse
import cv2
#python ./scripts/gom2.py --input_dir data/SRRAW_text --output_dir data/SRRAW_textEdit --image_file_extension '*jpg' --delete_specific_file_extension TRUE --delete_file_extension '*ARW'
image_size = 96

def main(args) -> None:
    delete_specific_file_extension = args.delete_specific_file_extension
    delete_file_extension = args.delete_file_extension
    input_dir = args.input_dir
    image_file_extension = args.image_file_extension
    output_dir = args.output_dir

    list_folder_input = [x[0] for x in os.walk(input_dir)]
    list_folder_input.sort(key=natural_keys)
    # print(list_folder_input)
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
        # print('Folder: ', folder_input)
        # Delete Specific file extention
        if delete_specific_file_extension:
            delete_file_names = glob(os.path.join(folder_input, delete_file_extension))
            for a in delete_file_names:
                os.remove(a)
        # Get all image paths
        image_file_names = glob(os.path.join(folder_input, image_file_extension))
        # print('List: ', image_file_names)
        print('Length Input {dir}: '.format(dir=folder_input), len(image_file_names))
        for image in image_file_names:
            img = cv2.imread(image)
            [h, w, _] = img.shape
            if h >= image_size and w >= image_size:
                worker(folder_input, image, output_dir, index)
                index += 1


def worker(images_dir, image_file_name, output_dir, index) -> None:
    new_image_name = f"{index}.{image_file_name.split('.')[-1]}"
    shutil.copyfile(f"{image_file_name}", f"{output_dir}/{new_image_name}")

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
    parser = argparse.ArgumentParser(description="Prepare database scripts.")
    parser.add_argument("--input_dir", type=str, help="Path to input image directory.")
    parser.add_argument("--output_dir", type=str, help="Path to generator image directory.")
    parser.add_argument("--image_file_extension", type=str, help="File extension of input image.")
    parser.add_argument("--delete_specific_file_extension", type=bool, default=False,
                        help="If True, all file with folow file extension will be delete.")
    parser.add_argument("--delete_file_extension", type=str, default=None, help="File extension need to delete.")
    args = parser.parse_args()

    main(args)

