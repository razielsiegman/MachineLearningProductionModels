# MNIST Digit Recognition Model

This is a productionized inference system machine learning model for the MNIST dataset. The model is able to recognize handwritten digits that are numerically represented by a one-dimensional array of length 784.

## Model Description
The model is built using the Sequential model architecture from the Keras library. It contains four layers, where the first layer is an input layer with 784 neurons. The second layer has 392 neurons, the third layer has 196 neurons, and the final output layer has 10 neurons. Each layer, except for the output layer, uses the ReLU activation function, while the output layer uses the Softmax activation function. The model uses the Stochastic Gradient Descent (SGD) optimizer with a learning rate of 0.1 and the categorical cross-entropy loss function.

## Running the Model
To run the model, simply run the file "production_model.py", which will prompt you to enter a file that contains a handwritten number, numerically represented by a one-dimensional array of length 784. The input should have integers with a max value of 255. To quit, enter Quit.

## Dependencies
This model requires the following dependencies:

numpy
keras
sklearn

## Example Usage
```
python mnist_inference_system.py
```
Please enter a file. The file should contain a handwritten number, numerically represented by a one dimensional array of length 784. To quit, enter "Quit":

```
/path/to/handwritten_digit.txt
```

Output:
```
Predicted Integer: 3
```

Please enter a file. The file should contain a handwritten number, numerically represented by a one dimensional array of length 784. To quit, enter "Quit":

```
Quit
```

Output:

```
Goodbye :)
```
