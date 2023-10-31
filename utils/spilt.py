import os
import random
import shutil
def spilt(source_dir:str, target_dir:str, percentage:int ):
    for class_dir in os.listdir(source_dir):
        class_path = os.path.join(source_dir, class_dir) #imgs/class
        if not os.path.isdir(class_path):
            continue
        
        target_class_path = os.path.join(target_dir, class_dir)#test/class
        os.makedirs(target_class_path, exist_ok=True)
        subdirs = [d for d in os.listdir(class_path) if os.path.isdir(os.path.join(class_path, d))]
        num_subdirs = len(subdirs)
        num_selected = int(num_subdirs * percentage)
        selected_subdirs = random.sample(subdirs, num_selected)
        
        for subdir in selected_subdirs:
            src_path = os.path.join(class_path, subdir)
            dst_path = os.path.join(target_class_path, subdir)
            shutil.move(src_path, dst_path)