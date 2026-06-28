import os
import random
import shutil

src_img_train = "data\\images\\train"
src_lbl_train = "data\\labels\\train"

src_img_val = "data\\images\\val"
src_lbl_val = "data\\labels\\val"

dst_base_dir = "visdrone_subset"

def copy_subset(src_img_dir, src_lbl_dir, dst_img_dir, dst_lbl_dir, count):
    # prepare destination directories
    os.makedirs(dst_img_dir, exist_ok=True)
    os.makedirs(dst_lbl_dir, exist_ok=True)
    
    # get all image files in the source directory
    all_images = [f for f in os.listdir(src_img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if len(all_images) < count:
        print(f"alert: directory {src_img_dir} contains only {len(all_images)} images.")
        count = len(all_images)
        
    # select a random subset of images
    selected_images = random.sample(all_images, count)
    
    copied_count = 0
    for img_name in selected_images:
        # copy image file
        shutil.copy(os.path.join(src_img_dir, img_name), os.path.join(dst_img_dir, img_name))
        
        # get corresponding label file name
        base_name = os.path.splitext(img_name)[0]
        lbl_name = base_name + ".txt"
        
        src_lbl_path = os.path.join(src_lbl_dir, lbl_name)
        
        # copy label file if it exists
        if os.path.exists(src_lbl_path):
            shutil.copy(src_lbl_path, os.path.join(dst_lbl_dir, lbl_name))
            copied_count += 1
        else:
            print(f"alert: file not found for image {img_name}")

    print(f"alert: copied {len(selected_images)} images and {copied_count} label files to {dst_img_dir}")

# alert: running script for train
copy_subset(
    src_img_train, src_lbl_train, 
    os.path.join(dst_base_dir, "images/train"), 
    os.path.join(dst_base_dir, "labels/train"), 
    500
)

# alert: running script for val
copy_subset(
    src_img_val, src_lbl_val, 
    os.path.join(dst_base_dir, "images/val"), 
    os.path.join(dst_base_dir, "labels/val"), 
    100
)
