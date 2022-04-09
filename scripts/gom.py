import os
import shutil


output_dir = '/home/minhhuy/Desktop/Python/SRGAN_DACN/data/Textdata/test'
image_dirs = ['/home/minhhuy/Desktop/Python/SRGAN_DACN/data/Textdata/datasets/Test30'
]
num_workers = 1

def main() -> None:
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    for images_dir in image_dirs:
        # Get all image paths
        image_file_names = os.listdir(images_dir)
        for image in image_file_names:
            worker(images_dir, image)


def worker(images_dir, image_file_name) -> None:
    folder_name = images_dir.split('/')[-1]
    new_image_name = f"{folder_name}_{image_file_name}"
    shutil.copyfile(f"{images_dir}/{image_file_name}", f"{output_dir}/{new_image_name}")


if __name__ == "__main__":
    main()
