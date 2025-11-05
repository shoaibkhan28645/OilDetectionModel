import os
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

class OilDataPreprocessor:
    def __init__(self, img_size=(224, 224), batch_size=32, classes=None, preprocess_fn=None):
        self.img_size = img_size
        self.batch_size = batch_size
        self.classes = classes
        self.preprocess_fn = preprocess_fn

    def _infer_classes(self, directory):
        """Infer class names from sub-directories."""
        if not os.path.exists(directory):
            raise FileNotFoundError(f"Directory {directory} does not exist")

        classes = [
            d for d in sorted(os.listdir(directory))
            if os.path.isdir(os.path.join(directory, d))
        ]

        if not classes:
            raise ValueError(
                f"No class sub-directories found in {directory}. "
                "Please create one sub-directory per oil type."
            )

        return classes
        
    def load_and_preprocess_image(self, image_path):
        """Load and preprocess a single image"""
        image = cv2.imread(image_path)
        if image is None:
            return None
            
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, self.img_size)
        image = image.astype(np.float32) / 255.0
        return image
    
    def load_dataset_from_directory(self, data_dir, classes=None):
        """Load dataset from directory structure"""
        if classes is None:
            classes = self.classes or self._infer_classes(data_dir)

        self.classes = classes

        images = []
        labels = []
        
        for class_idx, class_name in enumerate(classes):
            class_path = os.path.join(data_dir, class_name)
            if not os.path.exists(class_path):
                print(f"Warning: Directory {class_path} does not exist")
                continue
                
            for filename in os.listdir(class_path):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_path = os.path.join(class_path, filename)
                    image = self.load_and_preprocess_image(image_path)
                    
                    if image is not None:
                        images.append(image)
                        labels.append(class_idx)
        
        return np.array(images), np.array(labels), classes
    
    def create_data_generators(self, train_dir, validation_dir, classes=None):
        """Create data generators with augmentation for training"""

        if classes is None:
            classes = self.classes or self._infer_classes(train_dir)

        self.classes = classes

        if len(classes) <= 1:
            raise ValueError(
                f"Expected at least two classes for training, found {len(classes)} "
                f"in {train_dir}"
            )

        class_mode = 'binary' if len(classes) == 2 else 'categorical'
        
        train_datagen = ImageDataGenerator(
            rescale=None if self.preprocess_fn else 1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            brightness_range=[0.8, 1.2],
            fill_mode='nearest',
            preprocessing_function=self.preprocess_fn
        )
        
        validation_datagen = ImageDataGenerator(
            rescale=None if self.preprocess_fn else 1./255,
            preprocessing_function=self.preprocess_fn
        )
        
        train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode=class_mode,
            classes=classes,
            shuffle=True
        )
        
        validation_generator = validation_datagen.flow_from_directory(
            validation_dir,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode=class_mode,
            classes=classes,
            shuffle=False
        )
        
        return train_generator, validation_generator, classes
    
    def visualize_samples(self, data_dir, num_samples=8):
        """Visualize sample images from each class"""
        fig, axes = plt.subplots(2, num_samples//2, figsize=(15, 6))
        fig.suptitle('Sample Images from Dataset')
        
        for class_idx, class_name in enumerate(self.classes):
            class_path = os.path.join(data_dir, class_name)
            if not os.path.exists(class_path):
                continue
                
            images = [f for f in os.listdir(class_path) 
                     if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            for i in range(min(num_samples//2, len(images))):
                image_path = os.path.join(class_path, images[i])
                image = self.load_and_preprocess_image(image_path)
                
                if image is not None:
                    axes[class_idx, i].imshow(image)
                    axes[class_idx, i].set_title(f'{class_name}')
                    axes[class_idx, i].axis('off')
        
        plt.tight_layout()
        plt.show()

def prepare_data_split(source_dir, train_dir, val_dir, val_split=0.2, classes=None):
    """Split data into training and validation sets"""
    
    if classes is None:
        classes = [
            d for d in os.listdir(source_dir)
            if os.path.isdir(os.path.join(source_dir, d))
        ]

    if not classes:
        print(f"No class directories found in {source_dir}")
        return

    for class_name in classes:
        source_class_dir = os.path.join(source_dir, class_name)
        
        if not os.path.exists(source_class_dir):
            print(f"Source directory {source_class_dir} does not exist")
            continue
            
        images = [f for f in os.listdir(source_class_dir) 
                 if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        if len(images) == 0:
            print(f"No images found in {source_class_dir}")
            continue
            
        train_images, val_images = train_test_split(
            images, test_size=val_split, random_state=42
        )
        
        train_class_dir = os.path.join(train_dir, class_name)
        val_class_dir = os.path.join(val_dir, class_name)
        
        os.makedirs(train_class_dir, exist_ok=True)
        os.makedirs(val_class_dir, exist_ok=True)
        
        for img in train_images:
            src = os.path.join(source_class_dir, img)
            dst = os.path.join(train_class_dir, img)
            if not os.path.exists(dst):
                import shutil
                shutil.copy2(src, dst)
        
        for img in val_images:
            src = os.path.join(source_class_dir, img)
            dst = os.path.join(val_class_dir, img)
            if not os.path.exists(dst):
                import shutil
                shutil.copy2(src, dst)
        
        print(f"{class_name}: {len(train_images)} training, {len(val_images)} validation images")

if __name__ == "__main__":
    preprocessor = OilDataPreprocessor()
    
    data_dir = "../data/train"
    if os.path.exists(data_dir):
        preprocessor.visualize_samples(data_dir)
    else:
        print("Please add your images to the data/train directory first!")
        print("Structure should be:")
        print("data/train/coriander_oil/")
        print("data/train/mustard_oil/")
