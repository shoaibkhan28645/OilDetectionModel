# Oil Detection Model

A machine learning project to classify different types of oil (Coriander Oil vs Mustard Oil) applied on hands using computer vision and deep learning techniques.

## Project Overview

This project implements a binary image classifier that can distinguish between:
- **Coriander Oil** applied on hands
- **Mustard Oil** applied on hands

The model uses transfer learning with MobileNetV2 for efficient and accurate classification, making it suitable for university projects and real-world applications.

## Features

- 🤖 **Transfer Learning**: Uses pre-trained MobileNetV2 for better performance
- 📊 **Data Augmentation**: Automatically augments training data for better generalization
- 📈 **Training Visualization**: Plots training history and confusion matrices
- 🔍 **Batch Prediction**: Can predict on single images or entire directories
- 💾 **Model Persistence**: Save and load trained models
- 📱 **Lightweight**: Optimized for deployment on various platforms

## Project Structure

```
OilDetectionModel/
├── data/
│   ├── train/
│   │   ├── coriander_oil/       # Training images of coriander oil
│   │   └── mustard_oil/         # Training images of mustard oil
│   ├── validation/
│   │   ├── coriander_oil/       # Validation images of coriander oil
│   │   └── mustard_oil/         # Validation images of mustard oil
│   └── test/                    # Test images for inference
├── src/
│   ├── data_preprocessing.py    # Data loading and preprocessing
│   ├── model.py                 # Model architecture definitions
│   ├── train.py                 # Training script
│   └── predict.py               # Prediction/inference script
├── models/                      # Saved trained models
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## Installation

1. **Clone or download this project**

2. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare your dataset**:
   - Add coriander oil images to `data/train/coriander_oil/`
   - Add mustard oil images to `data/train/mustard_oil/`
   - Add validation images to respective validation folders

## Usage

### 1. Training the Model

Navigate to the `src/` directory and run:

```bash
# Train with transfer learning (recommended)
python train.py --transfer

# Train simple CNN from scratch
python train.py --simple
```

The training script will:
- Load and preprocess your images
- Create data augmentation pipeline
- Train the model with early stopping
- Save the best model automatically
- Display training plots and evaluation metrics

### 2. Making Predictions

After training, use the saved model for predictions:

```bash
# Predict on a single image
python predict.py ../models/best_oil_detection_model.h5 path/to/your/image.jpg

# Predict on all images in a directory
python predict.py ../models/best_oil_detection_model.h5 ../data/test/
```

### 3. Testing the Setup

You can test the model architecture without training:

```bash
python model.py  # Shows model summary and parameters
```

## Data Requirements

### Image Guidelines
- **Format**: JPG, PNG, or JPEG
- **Size**: Any size (automatically resized to 224x224)
- **Content**: Clear images of oil applied on hands
- **Quantity**: Minimum 50 images per class, recommended 100+ per class

### Directory Structure
Place your images in the following structure:
```
data/
├── train/
│   ├── coriander_oil/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   └── mustard_oil/
│       ├── image1.jpg
│       ├── image2.jpg
│       └── ...
└── validation/
    ├── coriander_oil/
    └── mustard_oil/
```

## Model Architecture

The project offers two model options:

### 1. Transfer Learning (Recommended)
- Base: MobileNetV2 pre-trained on ImageNet
- Custom classifier head with Global Average Pooling
- Dropout layer for regularization
- Binary classification output

### 2. Simple CNN
- 4 Convolutional layers with MaxPooling
- Dropout regularization
- Dense layers for classification
- Suitable for smaller datasets

## Training Parameters

- **Image Size**: 224x224 pixels
- **Batch Size**: 16-32 (adjustable)
- **Epochs**: 25-50 (with early stopping)
- **Optimizer**: Adam
- **Loss Function**: Binary Crossentropy
- **Data Augmentation**: Rotation, zoom, brightness, horizontal flip

## Performance Metrics

The model tracks the following metrics:
- **Accuracy**: Overall classification accuracy
- **Loss**: Binary crossentropy loss
- **Precision & Recall**: Per-class performance
- **Confusion Matrix**: Detailed classification results

## Tips for Better Results

1. **Data Quality**:
   - Use high-quality, well-lit images
   - Ensure consistent hand positioning
   - Include variety in backgrounds and lighting

2. **Data Quantity**:
   - More data generally leads to better performance
   - Aim for balanced classes (equal number of images)

3. **Training**:
   - Use transfer learning for better results
   - Monitor validation metrics to avoid overfitting
   - Consider fine-tuning for specific improvements

## Troubleshooting

### Common Issues

1. **No images found error**:
   - Check directory structure matches the required format
   - Ensure image files have correct extensions (.jpg, .png, .jpeg)

2. **Out of memory errors**:
   - Reduce batch size in training script
   - Use smaller image sizes if needed

3. **Poor model performance**:
   - Check data quality and quantity
   - Ensure balanced dataset
   - Consider collecting more diverse images

### Getting Help

For university project questions:
1. Check that all dependencies are installed correctly
2. Verify your dataset structure matches the requirements
3. Start with a small dataset to test the pipeline
4. Review training logs for any error messages

## Technical Details

- **Framework**: TensorFlow 2.13+ with Keras
- **Backend**: Supports both CPU and GPU training
- **Image Processing**: OpenCV and PIL
- **Visualization**: Matplotlib and Seaborn
- **Model Format**: Keras HDF5 (.h5) format

## Example Results

After training, you should expect:
- **Training Accuracy**: 85-95% (depending on data quality)
- **Validation Accuracy**: 80-90% (with good generalization)
- **Inference Time**: <100ms per image on modern hardware

## Future Improvements

Potential enhancements for advanced users:
- Multi-class classification for more oil types
- Real-time video classification
- Mobile app deployment using TensorFlow Lite
- Data collection automation tools

---

**Note**: This project is designed for educational purposes and university assignments. The model performance depends heavily on the quality and quantity of training data collected.