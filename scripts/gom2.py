import shutil
import os
import glob
import re

# mydir = '/media/minhhuy/5E525A46525A22D7/Data/test'
# f = os.listdir(mydir)
# for i in f:
#     filelist = glob.glob(os.path.join(mydir,i,'*ARW'))
#     for a in filelist:
#         os.remove(a)

# mydir = '/media/minhhuy/5E525A46525A22D7/Data/test'
# filelist = glob.glob(os.path.join(mydir,'*JPG'))
# print(filelist)
# for a in filelist:
#     os.remove(a)


list_folder_input = os.listdir('/media/minhhuy/5E525A46525A22D7/Data/test/')

output_dir = '/media/minhhuy/5E525A46525A22D7/Data/test/'


def main() -> None:
    index = 1
    for folder_input in list_folder_input:
        # Get all image paths
        # image_file_names = glob.glob(os.path.join('/media/minhhuy/5E525A46525A22D7/Data/test/',folder_input,'*JPG'))
        # print('1 ',image_file_names)
        # print('Length Input {dir}: '.format(dir=folder_input), len(image_file_names))
        # for image in image_file_names:
        #     worker(os.path.join('/media/minhhuy/5E525A46525A22D7/Data/test/',folder_input), image, index)
        #     index += 1

        # image_file_names = glob.glob(os.path.join('/media/minhhuy/5E525A46525A22D7/Data/test/',folder_input,'aligned','*JPG'))
        # print('2 ',image_file_names)
        # print('Length Input {dir}: '.format(dir=folder_input), len(image_file_names))
        # for image in image_file_names:
        #     worker(os.path.join('/media/minhhuy/5E525A46525A22D7/Data/test/',folder_input,'aligned'), image, index)
        #     index += 1
        shutil.rmtree(os.path.join('/media/minhhuy/5E525A46525A22D7/Data/test/', folder_input))


def worker(images_dir, image_file_name, index) -> None:
    new_image_name = f"{index}.{image_file_name.split('.')[-1]}"
    shutil.copyfile(f"{image_file_name}", f"{output_dir}/{new_image_name}")

if __name__ == "__main__":
    main()