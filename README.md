# Dog-vs-Cat-Image-Classification-with-CNN-and-FastAPI
Dog vs Cat classifier using TensorFlow CNN and FastAPI for real-time predictions.
# Dog vs Cat Classification using CNN and FastAPI  

A deep learning project that classifies images of **dogs and cats** using a **Convolutional Neural Network (CNN)**.  
The trained model is deployed with a **FastAPI backend** for real-time predictions.  

---

## Table of Contents  

- [Project Overview](#project-overview)  
- [Dataset](#dataset)  
- [Model Architecture](#model-architecture)  
- [Training Process](#training-process)  
- [Model Performance](#model-performance)  
- [Sample Predictions](#sample-predictions)  
- [API Usage](#api-usage)  
- [Project Structure](#project-structure)  
 

---

##  Project Overview  

This project demonstrates an end-to-end ML pipeline:  

Preprocessing the Dogs vs Cats dataset with **TensorFlow**  
Training a **CNN model** for binary classification  
Model evaluation and saving  
Serving predictions through a **FastAPI REST API**  

---

## Dataset  

We used the **Dogs vs Cats dataset** available on Kaggle:  
 [Dog and Cat Classification Dataset (Kaggle)](https://www.kaggle.com/datasets/bhavikjikadara/dog-and-cat-classification-dataset)  

- Total: **25,000 images** (dogs + cats)  
- Train: **20,000 images**  
- Test: **5,000 images**  

Data loading example:  

```python
train_ds = keras.utils.image_dataset_from_directory(
    "data/dogs_vs_cats/train",
    labels="inferred",
    label_mode="int",
    batch_size=32,
    image_size=(256, 256)
)

validation_ds = keras.utils.image_dataset_from_directory(
    "data/dogs_vs_cats/test",
    labels="inferred",
    label_mode="int",
    batch_size=32,
    image_size=(256, 256)
)
