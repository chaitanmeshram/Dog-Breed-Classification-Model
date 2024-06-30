
# Dog Breed Classification Machine Learning Model (Dog Vision)

## Overview
This repository contains a machine learning model designed to classify dog breeds from images. Using a dataset sourced from Kaggle, the model leverages the power of deep learning to achieve high accuracy in breed identification. This project utilizes Python and several libraries including Pandas, Numpy, Matplotlib, and TensorFlow.

## Features
- **Data Preprocessing**: Efficient handling and preprocessing of image data.
- **Model Architecture**: Implementation of a convolutional neural network (CNN) using TensorFlow and Keras.
- **Training and Evaluation**: Training the model on the dataset and evaluating its performance.
- **Visualization**: Visualizing data distribution, model performance, and classification results with Matplotlib.

##Link To Model Used: https://www.kaggle.com/models/google/mobilenet-v2/tensorFlow2/130-224-classification/1?tfhub-redirect=true

## Dataset
The dataset used in this project is sourced from Kaggle and contains a variety of dog breed images. It is essential to download the dataset from Kaggle and place it in the appropriate directory before running the model.
##Link To Data set: https://www.kaggle.com/c/dog-breed-identification/data

## Installation
To run this project, you'll need to have Python installed along with several libraries. Follow the instructions below to set up the environment:

1. **Clone the repository**:
    ```sh
    git clone https://github.com/yourusername/dog-breed-classification.git
    cd dog-breed-classification
    ```

2. **Create a virtual environment**:
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. **Install dependencies**:
    ```sh
    pip install -r requirements.txt
    ```

## Usage
1. **Download the dataset**:
   - Obtain the dataset from Kaggle and place it in the `data/` directory.

2. **Preprocess the data**:
    ```sh
    python preprocess.py
    ```

3. **Train the model**:
    ```sh
    python train.py
    ```

4. **Evaluate the model**:
    ```sh
    python evaluate.py
    ```

5. **Classify new images**:
    ```sh
    python classify.py --image_path path/to/your/image.jpg
    ```

## Project Structure
- `data/`: Directory to store the dataset.
- `notebooks/`: Google colab for data exploration and model development.
- `src/`: Source code for data preprocessing, model training, and evaluation.
- `models/`: Directory to save trained models.
- `README.md`: Project documentation.

## Results
The model achieves an accuracy of 99.88% on the test set, demonstrating its ability to accurately classify dog breeds from images.


