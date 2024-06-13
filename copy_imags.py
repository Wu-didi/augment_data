# Function: copy and rename images from different folders to one folder
# Author: wudi


import os
import shutil
from glob import glob
from tqdm import tqdm

def copy_and_rename_images(source_dirs, destination_dir):
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)

    for source_dir in  tqdm(source_dirs):
        image_files = glob(os.path.join(source_dir, '*.jpg')) + glob(os.path.join(source_dir, '*.bmp'))  # 可根据需要添加其他图片格式
        for idx,image_file in enumerate(image_files):
            _, file_extension = os.path.splitext(image_file)
            new_filename = f"{os.path.basename(source_dir)}_{idx}{file_extension}"
            destination_path = os.path.join(destination_dir, new_filename)
            shutil.copy(image_file, destination_path)

if __name__ == "__main__":
    source_dirs = os.listdir(r'/mnt/d/code/ubuntu/新建文件夹/新建文件夹/char-crops/NG')  # 替换为实际的源目录路径
    print(source_dirs)
    source_dirs = [os.path.join(r'/mnt/d/code/ubuntu/新建文件夹/新建文件夹/char-crops/NG', x) for x in source_dirs]
    print(source_dirs)
    destination_dir = r'/mnt/d/code/ubuntu/新建文件夹/新建文件夹/char-crops/NG_all' # 保存路径
    copy_and_rename_images(source_dirs, destination_dir)
    print("Images copied and renamed successfully.")
