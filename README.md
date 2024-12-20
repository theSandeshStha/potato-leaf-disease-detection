# Potato Leaf Disease Classification Using TensorFlow and Flask

This project is a machine learning-based application for classifying potato leaf conditions into three categories: **Early Blight**, **Late Blight**, and **Healthy**. It leverages a pre-trained TensorFlow model to predict the class of a given potato leaf image and integrates Flask for the web interface.

---

## Table of Contents

- [Installation](#installation)
- [Project Features](#project-features)
- [Usage Instructions](#usage-instructions)
- [Environment Details](#environment-details)
- [Model Information](#model-information)

---

# Overview

The application processes images of potato leaves to determine their health status. Users can upload images via the web interface, and the application validates the input, preprocesses the image, and classifies it using a deep learning model. The output includes the predicted class and the confidence score.

## Installation

1. Clone the repository:

```bash
   git clone https://github.com/your-username/potato-leaf-classification.git
   cd potato-leaf-classification
```

2. Create and activate a conda environment:

```bash
conda create -n potato-arm python=3.9
conda activate potato-arm
```

3. Install Dependencies:

```bash
pip install -r requirements.txt
```

# Project Features

- Image Preprocessing: Automatically resizes and normalizes input images.
- Leaf Validation: Detects whether the uploaded image resembles a leaf.
- Disease Classification: Predicts whether the leaf is affected by Early Blight, Late Blight, or is Healthy.
- Confidence Scoring: Provides confidence percentages for predictions.
- User Interface: Offers an intuitive web interface for image upload and result display.

# Usage Instructions

1. Run the Flask Application: Start the application by running:

```bash
python app.py
```

- The application will be hosted at http://127.0.0.1:5000.

2. Upload an Image:

- Open the web application in your browser.
- Upload an image of a potato leaf.

3. View Predictions:

- The application will display the predicted class of the leaf and the confidence score.

## Environment Details

### Key Packages

The following Python packages were used in this project:

| Package       | Version   |
| ------------- | --------- |
| Flask         | 3.0.3     |
| TensorFlow    | 2.17.0    |
| NumPy         | 1.23.5    |
| Pillow        | 11.0.0    |
| OpenCV-Python | 4.10.0.84 |

These packages are listed in the `requirements.txt` file.

## Model Information

- The model is a convolutional neural network (CNN) saved as potato.h5.

- It is trained on labeled data consisting of potato leaf images categorized into:

  - **Early Blight**
  - **Late Blight**
  - **Healthy**

## Prediction

- The model processes 256x256 RGB images and outputs predictions for the above classes.
- Confidence scores indicate the likelihood of each class.

## Contributing

Contributions to this project are welcome! To contribute:

1. Fork the repository.
2. Create a new feature branch.
3. Submit a pull request with a clear explanation of the changes.

Please follow coding best practices, include detailed comments, and ensure any new features are tested.
