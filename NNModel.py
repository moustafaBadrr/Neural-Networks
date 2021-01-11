import numpy as np
import statistics as sd
from numpy import float64


def read_file(file_name):
    read_data_file = open(file_name, "r")
    content = read_data_file.read()
    content_list = content.splitlines()
    first_line = content_list[0].split()
    n_input_layer = int(first_line[0])
    n_hidden_layer = int(first_line[1])
    n_output_layer = int(first_line[2])
    n_training = int(content_list[1])
    data_set_x = []
    data_set_y = []
    for i in range(2, len(content_list)):
        data = content_list[i].split()
        temp_x = []
        temp_y = []
        for j in range(0, len(data) - n_output_layer):
            temp_x.append(float(data[j]))
        data_set_x.append(temp_x)
        for k in range(n_input_layer, len(data)):
            temp_y.append(float(data[k]))
        data_set_y.append(temp_y)
    read_data_file.close()

    return n_input_layer, n_hidden_layer, n_output_layer, n_training, data_set_x, data_set_y


def normalize_dataset(x):  # rescale the data to get values between 0 and 1
    number_of_features = int(x[0].shape[0])
    means = []
    standard_deviation = []
    for i in range(0, number_of_features):
        means.append(np.mean(x[:, i]))
        standard_deviation.append(sd.stdev(x[:, i]))
    for i in range(0, number_of_features):
        if standard_deviation[i] != 0:
            x[:, i] = (x[:, i] - means[i]) / standard_deviation[i]
    return x


def initialize_parameters(n_x, n_h, n_y):
    np.random.seed(1)
    w1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros((n_h, 1))
    w2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros((n_y, 1))
    parameters = {"W1": w1,
                  "b1": b1,
                  "W2": w2,
                  "b2": b2}
    return parameters


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def sigmoid_derivative(z):
    f = sigmoid(z)
    return f * (1 - f)


def forward(x, parameters):
    caches = []
    a = x
    L = len(parameters) // 2
    a_prev = a
    z = np.dot(parameters['W' + str(1)], a_prev) + parameters['b' + str(1)]
    a = sigmoid(z)
    cache = ((a_prev, parameters['W' + str(1)], parameters['b' + str(1)]), z)
    caches.append(cache)

    a_prev = a
    z = np.dot(parameters['W' + str(L)], a_prev) + parameters['b' + str(L)]
    cache = ((a_prev, parameters['W' + str(L)], parameters['b' + str(L)]), z)
    caches.append(cache)
    return z, caches


def compute_cost(al, y):
    m = y.shape[1]
    cost = (1 / m) * np.sum((al - y) ** 2)
    return cost


def backward(AL, Y, caches):
    grads = {}
    m = AL.shape[1]
    A1 = caches[1][0][0]
    A0 = caches[0][0][0]
    w2 = caches[1][0][1]
    z1 = caches[0][1]
    dz2 = AL-Y
    dw2 = np.dot(dz2, A1.T) / m
    db2 = np.sum(dz2, axis=1, keepdims=True) / m
    dz1 = np.dot(w2.T, dz2)*sigmoid_derivative(z1)
    dw1 = np.dot(dz1, A0.T) / m
    db1 = np.sum(dz1, axis=1, keepdims=True) / m

    grads["dW" + str(1)] = dw1
    grads["db" + str(1)] = db1
    grads["dW" + str(2)] = dw2
    grads["db" + str(2)] = db2

    return grads


def update_parameters(parameters, grads, learning_rate):
    L = len(parameters) // 2
    for l in range(L):
        parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - learning_rate * grads["dW" + str(l + 1)]
        parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - learning_rate * grads["db" + str(l + 1)]
    return parameters


def nn(x, y, n_input_layer, n_hidden_layer, n_output_layer, iteration_num, learning_rate):
    parameters = initialize_parameters(n_input_layer, n_hidden_layer, n_output_layer)
    for i in range(iteration_num):
        a, caches = forward(x, parameters)

        print("Cost will be : " + str(compute_cost(a, y)))
        grads = backward(a, y, caches)
        parameters = update_parameters(parameters, grads, learning_rate)

    return parameters


def prediction(dat_set_x, parameters, data_set_y):
    a, caches = forward(dat_set_x, parameters)
    print("Cost will be : " + str(compute_cost(a, data_set_y)))


def main():
    n_input_layer, n_hidden_layer, n_output_layer, n_training, data_set_x, data_set_y = read_file("train.txt")

    x = np.array(data_set_x)
    y = np.array(data_set_y).T
    # normalize_dataset(y.T)
    x = normalize_dataset(x).T
    parameters = nn(x, y, n_input_layer, n_hidden_layer, n_output_layer, 1000, .1)
    prediction(x, parameters, y)


main()
