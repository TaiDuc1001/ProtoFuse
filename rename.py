import os
import shutil

os.makedirs('datasets/cub-200-2011-renamed/', exist_ok=True)
root_dir = 'datasets/CUB_200_2011/images'
target_dir = 'datasets/cub-200-2011-renamed/'
for dir in os.listdir(root_dir):
    src_dir = os.path.join(root_dir, dir)
    dest_dir = os.path.join(target_dir, dir.split('.')[-1])
    # print(src_dir, '->', dest_dir)
    if os.path.isdir(src_dir):
        shutil.copytree(src_dir, dest_dir, dirs_exist_ok=True)
        print(f'{src_dir} -> {dest_dir}')