# 🖼️ Computer Vision Practical Exam Project 

## Image Classification using Pretrained Deep Learning Models

This project is a complete **Image Classification pipeline** built using **TensorFlow / Keras** and a **pretrained CNN model** (e.g., VGG19, MobileNetV2, EfficientNetB0).  
The goal is to classify images into multiple categories using **Transfer Learning**, which improves accuracy and reduces training time.

---

## 🚀 Project Features

- ✔️ Uses a **pretrained CNN model** (ImageNet weights)
- ✔️ Transfer Learning + Fine-tuning
- ✔️ Data preprocessing & augmentation
- ✔️ Training with checkpoints & early stopping
- ✔️ Visualizing **training & validation accuracy/loss**
- ✔️ Saving the trained model
- ✔️ Inference script to test with new images

---


## 🧠 Pretrained Model Used

You can choose from:

- **VGG19**
- **EfficientNetB0**
- **MobileNetV2**
- **Custom CNN**

This project uses:

```python
applications.EfficientNetB0(weights="imagenet", include_top=False)