import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.datasets import mnist
from sklearn.model_selection import train_test_split
from keras.optimizers import SGD
from keras.utils import to_categorical


def get_model():
    (x_train_original, y_train_original), (x_validation_original, y_validation_original) = mnist.load_data()

    #data reshaping
    x_train_original = x_train_original.reshape((x_train_original.shape[0], 28*28)).astype('float32')
    x_validation = x_validation_original.reshape((x_validation_original.shape[0], 28*28)).astype('float32')

    #normalization
    x_train_original = ((x_train_original / 255.)-.5)*2
    x_validation = ((x_validation / 255.)-.5)*2

    #one-hot encoding
    y_train = to_categorical(y_train_original)
    y_val = to_categorical(y_validation_original)

    #from the training set, create a train and test set.
    x_train, x_test, y_train, y_test = train_test_split(x_train_original, y_train, test_size=0.2, random_state = 42)

    model = Sequential()
    model.add(Dense(28*14, activation = 'relu', input_dim=784))
    model.add(Dense(14*14, activation = 'relu'))
    model.add(Dense(7*7, activation = 'relu'))
    model.add(Dense(10, activation = 'softmax'))

    sgd_opt = SGD(learning_rate=.1, name="SGD")

    model.compile(optimizer=sgd_opt, loss='categorical_crossentropy', metrics = ['accuracy'])

    model.fit(x=x_train, y=y_train, epochs=4, batch_size = 100, verbose=0, validation_data=(x_test, y_test))

    return model

#Ensure that the input has 784 integers with a max value of 255
def valid_input(data):
    data = data.replace("\n", "")
    list = data.split()
    if len(list) != 784:
        return False
    return all([(item.isnumeric() and int(item) < 256) for item in list])

#Transform the string of whitespaced integers to an array, as well as normalization of the data
def format_data(data):
    list = data.split()
    int_list = [int(i) for i in list]
    int_array = np.array(int_list)
    normalized = ((int_array / 255.)-.5)*2
    normalized = np.reshape(normalized, (1, 784))

    return normalized

#Return the predicted integer
def evaluate(model, formatted_data):
    predictions = np.argmax(model.predict(formatted_data), axis=1)
    print('Precited Integer:', predictions)


def run(model):
    while True:
        file_name = input("\nPlease enter a file.  The file should contain a handwritten number, "
                     "numerically represented by a one dimensional array of length 784.  "
                     "To quit, enter \"Quit\":\n")
        if 'Quit' == file_name:
            break
        file = open(file_name, "r")
        data = file.read()
        if valid_input(data) == False:
            print('Input is invalid')
        else:
            formatted_data = format_data(data)
            evaluate(model, formatted_data)
    print('Goodbye :)')


model = get_model()
run(model)
