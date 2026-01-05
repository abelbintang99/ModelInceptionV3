# Image Classification Using CNN (InceptionV3)

## Project Description

This project aims to perform **image classification** using a **Convolutional Neural Network (CNN)** with the **InceptionV3** architecture. The model is developed to classify images into multiple classes based on the available image dataset.

To improve performance and reduce overfitting, the project applies:

* **Transfer Learning (InceptionV3 pretrained on ImageNet)**
* **Fine-Tuning**
* **K-Fold Cross Validation (5-Fold)**
* **Data Augmentation**

This project is conducted as part of a **final project / Machine Learning / Deep Learning course assignment**.

---

## Key Features

* Multi-class image classification
* CNN architecture using InceptionV3
* K-Fold Cross Validation (5 Folds)
* Model evaluation using:

  * Accuracy
  * Precision, Recall, F1-Score
  * Confusion Matrix
* Saving the best model from each fold

---

## Technologies Used

* Python 3.x
* TensorFlow & Keras
* NumPy
* Pandas
* Scikit-learn
* Matplotlib & Seaborn

---

## Model Configuration

* **Model**: InceptionV3
* **Input Size**: 299 x 299
* **Batch Size**: 32
* **Optimizer**: Adam
* **Loss Function**: Categorical Crossentropy
* **Epochs**: Adjusted with Early Stopping
* **K-Fold**: 5 Folds (Stratified)

---

## How to Run the Project

### 1. Clone the Repository

```bash
git clone https://github.com/username/klasifikasi-gambar-inceptionv3.git
cd klasifikasi-gambar-inceptionv3
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Prepare the Dataset

Ensure that the dataset is properly organized inside the `dataset/` directory according to each class structure.

### 4. Model Training

Run the training notebook or script:

```bash
jupyter notebook notebooks/training_inceptionv3.ipynb
```

### 5. Model Evaluation

The model is evaluated using the test dataset and produces a **classification report** and **confusion matrix**.

---

## Results and Evaluation

The best model is selected based on the **highest validation accuracy** in each fold. Final evaluation is performed using the test dataset to measure the model’s generalization performance.

Evaluation metrics include:

* Accuracy
* Precision
* Recall
* F1-Score
* Confusion Matrix

---

## Conclusion

The application of InceptionV3 with K-Fold Cross Validation effectively improves the stability and performance of the image classification model. Data augmentation and fine-tuning play an important role in reducing overfitting.

---

## Future Work

* Experimenting with other architectures (ResNet, EfficientNet)
* Further hyperparameter tuning
* Deployment implementation (Web / API)


## Author

**Abel Bintang**
Informatics Engineering

---

## License

This project is created for academic and educational purposes.
