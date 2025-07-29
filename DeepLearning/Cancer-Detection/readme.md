# Histopathologic Cancer Detection - Deep Learning Project

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![Jupyter](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=flat&logo=jupyter&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=flat&logo=TensorFlow&logoColor=white)
![Kaggle](https://img.shields.io/badge/Kaggle-035a7d?style=flat&logo=kaggle&logoColor=white)

## 🎯 Project Overview

This project tackles the **Histopathologic Cancer Detection** challenge from Kaggle - a binary image classification problem to identify metastatic cancer in small image patches taken from larger digital pathology scans. The goal is to build a deep learning model that can accurately distinguish between cancerous and non-cancerous tissue samples.

**Competition Link**: [Histopathologic Cancer Detection](https://www.kaggle.com/c/histopathologic-cancer-detection)

### 🏥 Medical Context
- **Task**: Binary classification of 96×96 pixel histopathologic images
- **Target**: Detect metastatic tissue in lymph node sections
- **Clinical Impact**: Early detection of cancer spread to lymph nodes
- **Evaluation Metric**: Area Under the ROC Curve (AUC)

## 📊 Results Summary

| Model | Validation AUC | Accuracy | Parameters | Training Time |
|-------|---------------|----------|------------|---------------|
| **EfficientNetB0** ⭐ | **0.931** | **89.1%** | 5.3M | 65 min |
| ResNet50 | 0.923 | 88.6% | 25.6M | 78 min |
| DenseNet121 | 0.919 | 88.3% | 8.1M | 82 min |
| Custom CNN | 0.847 | 82.1% | 2.1M | 45 min |
| **Ensemble** | **0.938** | **89.5%** | 39.0M | 125 min |

**🏆 Best Performance**: EfficientNetB0 achieved **0.931 AUC** with optimal efficiency-performance trade-off.


## 🔬 Methodology

### 1. Exploratory Data Analysis (EDA)
- **Dataset**: 220,025 training images, class distribution analysis
- **Image Properties**: 96×96 pixels, RGB format, .tif files
- **Class Balance**: Analyzed cancer vs. non-cancer distribution
- **Data Quality**: Verified image integrity and label consistency

### 2. Model Architecture Comparison

#### **Custom CNN**
- Lightweight architecture designed for 96×96 medical images
- Progressive feature extraction: 32→64→128→256 filters
- Batch normalization and dropout for regularization

#### **Transfer Learning Models**
- **EfficientNetB0**: Compound scaling, optimal efficiency
- **ResNet50**: Deep residual connections, proven performance  
- **DenseNet121**: Dense connections for feature reuse

#### **Ensemble Method**
- Weighted combination of top 3 models
- 50% EfficientNetB0 + 30% ResNet50 + 20% DenseNet121

### 3. Hyperparameter Optimization
- **Learning Rate**: [0.0001, 0.0005, 0.001] → Optimal: 0.0005
- **Batch Size**: [16, 32, 64] → Optimal: 32
- **Optimizer**: Adam vs RMSprop → Adam performed better
- **Dropout**: [0.2, 0.3, 0.5] → Optimal: 0.3-0.4

### 4. Training Strategy
- **Phase 1**: Frozen backbone training (5 epochs)
- **Phase 2**: Partial unfreezing with reduced LR (10 epochs)
- **Phase 3**: Full fine-tuning with minimal LR (5 epochs)

## 📈 Key Findings

### ✅ What Worked Well

1. **Transfer Learning Superiority** (10% improvement)
   - ImageNet features transferable to medical imaging
   - Significantly outperformed custom CNN

2. **EfficientNet Architecture** 
   - Best efficiency-accuracy trade-off
   - Lower parameter count than ResNet50 with better performance

3. **Data Augmentation**
   - Rotation, flipping crucial for pathology images
   - Brightness adjustment for staining variations

4. **Class Weight Balancing**
   - Critical for medical applications
   - Improved sensitivity from 82% to 91%

## 🛠️ Technical Implementation

### Dependencies
```python
tensorflow>=2.8.0
scikit-learn>=1.0.0
numpy>=1.21.0
pandas>=1.4.0
matplotlib>=3.5.0
pillow>=8.0.0
seaborn>=0.11.0
```

### Key Code Snippets

**Data Augmentation**:
```python
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    horizontal_flip=True,
    vertical_flip=True,
    brightness_range=[0.8, 1.2],
    zoom_range=0.1
)
```

**Best Model Configuration**:
```python
optimal_config = {
    'architecture': 'EfficientNetB0',
    'learning_rate': 0.0005,
    'batch_size': 32,
    'optimizer': 'Adam',
    'dropout_rates': [0.3, 0.2],
    'class_weights': {0: 0.57, 1: 1.75}
}
```

## 📊 Performance Analysis

### Training Curves
- Smooth convergence without overfitting
- Validation AUC reached 0.931 after 20 epochs
- Good generalization (train-val gap < 0.05)

### Medical AI Considerations
- **False Negative Impact**: Prioritized sensitivity for cancer detection
- **Class Imbalance**: Handled with balanced class weights
- **Interpretability**: Future work on attention visualization needed



```bibtex
@misc{cancer_detection_2024,
  title={Histopathologic Cancer Detection using Deep Learning},
  author={Herman Justino},
  year={2024},
  institution={MSc Data Science Program},
  note={Kaggle Competition Implementation}
}
```

## 📞 Contact

- **Author**: Herman Justino
- **Program**: MSc Data Science
- **Course**: Deep Learning
- **Competition**: [Kaggle Histopathologic Cancer Detection](https://www.kaggle.com/c/histopathologic-cancer-detection)

---

## 🔗 Links

- **Jupyter Notebook**: [`notebooks/index.ipynb`](notebooks/index.ipynb)
- **Kaggle Competition**: [Histopathologic Cancer Detection](https://www.kaggle.com/c/histopathologic-cancer-detection)
- **Dataset**: Download from Kaggle competition page

---

*This project demonstrates the application of modern deep learning techniques to medical imaging, achieving clinically relevant performance while maintaining focus on educational objectives and methodological rigor.*
