# 🍠 Potato Disease Classification using CNN
## Objective:
The main goal of this project is to classify different diseases in potato plants based on leaf images. This can assist farmers and agricultural experts in diagnosing and treating potato diseases more efficiently.

## Model:
I built a CNN (Convolutional Neural Network) to process images and classify them into different categories of diseases. CNNs are particularly well-suited for image classification tasks because they automatically detect important features such as edges, textures, and shapes from the images. In this case, the model categorizes potato leaf images into 3 classes: healthy and two other disease types.

## Technologies Used:
- TensorFlow and Keras were used to develop the CNN model.
- FastAPI was employed to create an API, enabling users to send images and receive disease predictions.
- NumPy was used for manipulating the images as arrays before feeding them into the model.
- Matplotlib helped visualize the data, such as model performance and potentially image outputs during testing or debugging.

## Project Workflow
### Dataset Preparation:
I used a dataset of potato leaf images, which are pre-labeled with the correct disease category. These images were processed and augmented to increase training data diversity, making the model more robust to variations in real-world images (like different lighting or angles).

### CNN Model:
The CNN architecture consists of multiple Conv2D layers to extract features from the input image, each followed by MaxPooling to downsample the feature maps and retain the most important information. After several convolution and pooling layers, the data is flattened and passed through Dense layers for classification. The final layer uses a softmax activation function, which outputs a probability distribution across the different classes (3 in this case).

### Training:
The CNN model was trained on the labeled dataset using backpropagation and optimization algorithms (e.g., Adam or SGD). During training, the model learned to minimize the error in predicting the correct disease class based on the input images.

### Serving the Model:
After training, the model was saved as a TensorFlow SavedModel (saved_model.pb) file. I loaded this saved model during inference in the FastAPI backend.

### API Functionality:
I developed an API that allows users to upload an image of a potato leaf via a POST request to the /predict/ endpoint. The uploaded image is processed (resized and scaled), and the CNN model predicts the disease. The API responds with the predicted class, such as healthy or a specific disease type.

### Project Structure:
Model: The core CNN model, trained on potato disease image data.
API: A RESTful API using FastAPI, allowing users to upload images and receive predictions.
Dependencies: Various Python libraries such as TensorFlow, FastAPI, Pillow, NumPy, and others for deep learning and API development.