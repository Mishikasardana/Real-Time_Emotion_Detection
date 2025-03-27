# Real-Time_Emotion_Detection
# Image Classification Project

This project demonstrates an image classification task using a pre-trained MobileNet model in Keras/TensorFlow. The goal is to classify images into different categories (in this case, 7 categories as indicated by the final dense layer).

## Setup

1.  **Download the Dataset:**
    The training dataset is downloaded as a zip file (`train.zip`) from a Dropbox link.
    ```bash
    !wget -O train.zip "[https://www.dropbox.com/scl/fi/qtcrzmsv4jz47cgvr2t0u/train.zip?rlkey=x3sz1h2o797inkqer67q7dg6b&dl=1](https://www.dropbox.com/scl/fi/qtcrzmsv4jz47cgvr2t0u/train.zip?rlkey=x3sz1h2o797inkqer67q7dg6b&dl=1)"
    ```

2.  **Unzip the Dataset:**
    The downloaded zip file is extracted to access the image data.
    ```bash
    !unzip train.zip
    ```
    The images are expected to be organized into subdirectories within the `/content/train` directory, where each subdirectory represents a different class.

3.  **Install Libraries:**
    Ensure you have the necessary libraries installed. If you're running this in a Colab environment, most of these are pre-installed. Otherwise, you might need to install them using pip:
    ```bash
    pip install numpy pandas matplotlib tensorflow
    ```

## Model Architecture

-   **Base Model:** A pre-trained MobileNet model is used as the base for feature extraction. The top classification layer of MobileNet is excluded (`include_top=False`).
-   **Transfer Learning:** The layers of the pre-trained MobileNet model are frozen (`layer.trainable = False`) to leverage the learned features without significantly altering them during training.
-   **Custom Classification Head:** A Flatten layer and a Dense output layer with 7 units and a softmax activation function are added on top of the MobileNet base. The 7 units correspond to the number of classes in the dataset.

## Training

1.  **Data Augmentation:** An `ImageDataGenerator` is used to augment the training data. This includes:
    -   Zooming (`zoom_range=0.2`)
    -   Shearing (`shear_range=0.2`)
    -   Horizontal flipping (`horizontal_flip=True`)
    -   Rescaling pixel values to the range \[0, 1] (`rescale = 1./255`).

2.  **Training Data Generator:** A `flow_from_directory` method is used to load images from the `/content/train` directory, resizing them to 224x224 pixels and creating batches of 32 images.

3.  **Validation Data Generator:** A separate `ImageDataGenerator` (with only rescaling) and `flow_from_directory` are used to load validation data from the same `/content/train` directory for monitoring performance during training. **Note:** In a real-world scenario, you would typically have a separate directory for validation data.

4.  **Model Compilation:** The model is compiled with the Adam optimizer, categorical cross-entropy loss (suitable for multi-class classification), and accuracy as the evaluation metric.

5.  **Callbacks:**
    -   **Early Stopping:** `EarlyStopping` is used to monitor the validation accuracy (`val_accuracy`) and stop training if it doesn't improve for 5 consecutive epochs (`patience=5`). This helps prevent overfitting.
    -   **Model Checkpoint:** `ModelCheckpoint` saves the best model weights (based on `val_accuracy`) to the file `best_model.h5`.

6.  **Model Fitting:** The model is trained using the `fit` method with the training and validation data generators, specifying the number of steps per epoch, total epochs, and the defined callbacks.

## Evaluation

-   The training and validation accuracy and loss are recorded during training and can be visualized.
-   The final training and validation accuracies are printed after training.
-   The best performing model (saved by the `ModelCheckpoint` callback) is loaded from `best_model.h5`.
-   Plots of training vs. validation accuracy and loss are generated to assess the model's learning progress and identify potential overfitting.

## Prediction

-   A sample image (`happy-girl-16.png`) is loaded, preprocessed, and used to demonstrate how to make predictions with the trained model.
-   The predicted class index is mapped back to the original class name using the `class_indices` from the training data generator.
-   The input image and its predicted class are displayed.

## Code Structure

The Python script performs the following main steps:

1.  Downloads and extracts the dataset.
2.  Builds the MobileNet-based classification model.
3.  Configures data generators for training and validation.
4.  Defines and uses callbacks for early stopping and model checkpointing.
5.  Trains the model.
6.  Evaluates the training history and final performance.
7.  Demonstrates prediction on a sample image.
8.  Loads the best model and visualizes training history.

## Potential Improvements

-   **Separate Validation Set:** Use a dedicated directory for validation data instead of using the training data directory for both.
-   **More Data Augmentation:** Experiment with a wider range of augmentation techniques to improve the model's generalization.
-   **Fine-tuning:** After initial training with frozen layers, consider unfreezing some of the top layers of the MobileNet model for fine-tuning on the specific task. This should be done with a lower learning rate.
-   **Hyperparameter Tuning:** Optimize the model's hyperparameters (e.g., learning rate, optimizer, batch size, number of dense units) using techniques like grid search or random search.
-   **Evaluation on Test Set:** After training and validation, evaluate the final best model on a completely unseen test dataset to get an unbiased estimate of its performance.
