# Hand-Writing-Detection
Used to understand and detect hand written digits

Using Modified National Institute of Standards and Technology MNIST for short which was developed by god father of CNN YANN LECUN I implemented this project. This project was done in 3 ways especially....
1. Using simple Artifical Neural Network
2. Using simple Convolutional Neural Network
3. Using Multi layered COnvolutional Neural Network

Among the 3 methods last method i.e., the one using Multi layer CNN gave better accuracy and less loss. 

The main reason that is select MNIST dataset is that in it the images of digits were taken from a variety of scanned documents, normalized in size and centered. This makes it an excellent dataset for evaluating models, allowing the developer to focus on the machine learning with very little data cleaning or preparation required.

Each image is a 28 by 28 pixel square (784 pixels total). A standard spit of the dataset is used to evaluate and compare models, where 60,000 images are used to train a model and a separate set of 10,000 images are used to test it.

It is a digit recognition task. As such there are 10 digits (0 to 9) or 10 classes to predict. Results are reported using prediction error, which is nothing more than the inverted classification accuracy.

Training using Simple multilayer ANN:

The training dataset is structured as a 3-dimensional array of instance, image width and image height. For a multi-layer perceptron model we must reduce the images down into a vector of pixels. In this case the 28×28 sized images will be 784 pixel input values

The output variable is an integer from 0 to 9. This is a multi-class classification problem. As such, it is good practice to use a one hot encoding of the class values, transforming the vector of class integers into a binary matrix

A softmax activation function is used on the output layer to turn the outputs into probability-like values and allow one class of the 10 to be selected as the model’s output prediction. Logarithmic loss is used as the loss function (called categorical_crossentropy in Keras) and the efficient ADAM gradient descent algorithm is used to learn the weights.


Trainig using Simple CNN:

We need to load the MNIST dataset and reshape it so that it is suitable for use training a CNN. In Keras, the layers used for two-dimensional convolutions expect pixel values with the dimensions [pixels][width][height].

In the case of RGB, the first dimension pixels would be 3 for the red, green and blue components and it would be like having 3 image inputs for every color image. In the case of MNIST where the pixel values are gray scale, the pixel dimension is set to 1


The first hidden layer is a convolutional layer called a Convolution2D. The layer has 32 feature maps, which with the size of 5×5 and a rectifier activation function. This is the input layer, expecting images with the structure outline above [pixels][width][height].

Next we define a pooling layer that takes the max called MaxPooling2D. It is configured with a pool size of 2×2.
The next layer is a regularization layer using dropout called Dropout. It is configured to randomly exclude 20% of neurons in the layer in order to reduce overfitting.

Next is a layer that converts the 2D matrix data to a vector called Flatten. It allows the output to be processed by standard fully connected layers.

Next a fully connected layer with 128 neurons and rectifier activation function.

Finally, the output layer has 10 neurons for the 10 classes and a softmax activation function to output probability-like predictions for each 
class.

As before, the model is trained using logarithmic loss and the ADAM gradient descent algorithm.


Training using Large CNN:

This time we define a large CNN architecture with additional convolutional, max pooling layers and fully connected layers. The network topology can be summarized as follows
	Convolutional layer with 30 feature maps of size 5×5.
	Pooling layer taking the max over 2*2 patches.
	Convolutional layer with 15 feature maps of size 3×3.
	Pooling layer taking the max over 2*2 patches.
	Dropout layer with a probability of 20%.
	Flatten layer.
	Fully connected layer with 128 neurons and rectifier activation.
	Fully connected layer with 50 neurons and rectifier activation.
	Output layer.
  
  
  Note the code for all the 3 types of implemention are available in this repo
