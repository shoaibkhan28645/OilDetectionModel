import os
import shutil
import random

# Split 20% of training images to validation
def quick_split():
    classes = ['coriander_oil', 'mustard_oil']
    
    for class_name in classes:
        train_dir = f"data/train/{class_name}"
        val_dir = f"data/validation/{class_name}"
        
        if not os.path.exists(train_dir):
            continue
            
        images = [f for f in os.listdir(train_dir) 
                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        # Take 20% for validation
        val_count = max(1, len(images) // 5)
        val_images = random.sample(images, val_count)
        
        for img in val_images:
            src = os.path.join(train_dir, img)
            dst = os.path.join(val_dir, img)
            shutil.move(src, dst)
        
        print(f"{class_name}: {len(images)-val_count} train, {val_count} validation")

if __name__ == "__main__":
    quick_split()