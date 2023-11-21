import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

class NeuralNetwork1(nn.Module):
    def __init__(self, input_dimension, output_dimension):
        super(NeuralNetwork1, self).__init__()
        
        # Lớp ẩn 1 với hàm kích hoạt ReLU
        self.layer_hidden1 = nn.Linear(input_dimension, 32)
        self.activation_relu1 = nn.ReLU()
        
        # Lớp ẩn 2 với hàm kích hoạt ReLU
        self.layer_hidden2 = nn.Linear(32, 64)
        self.activation_relu2 = nn.ReLU()
        
        # Lớp output với hàm kích hoạt softmax
        self.layer_output = nn.Linear(64, output_dimension)
        self.activation_softmax = nn.Softmax()

        # Khởi tạo trọng số ngẫu nhiên
        #self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.random_(0, 1)
            if module.bias is not None:
                module.bias.data.zero_()


    def forward(self, x):
        x = x.float()
        #print(x)
        x = self.activation_relu1(self.layer_hidden1(x))
        #print(x)
        x = self.activation_relu2(self.layer_hidden2(x))
        #print(x)
        #x = self.activation_relu3(self.layer_hidden3(x))
        x = self.activation_softmax(self.layer_output(x))
        #print(x)
        return x

class NeuralNetwork2(nn.Module):
    def __init__(self, input_dimension, output_dimension):
        super(NeuralNetwork2, self).__init__()
        
        # Lớp ẩn 1 với hàm kích hoạt ReLU
        self.layer_hidden1 = nn.Linear(input_dimension, 8)
        self.activation_relu1 = nn.ReLU()
        
        # Lớp ẩn 2 với hàm kích hoạt ReLU
        self.layer_hidden2 = nn.Linear(8, 16)
        self.activation_relu2 = nn.ReLU()
        
        # Lớp output với hàm kích hoạt softmax
        self.layer_output = nn.Linear(16, output_dimension)
        self.activation_softmax = nn.Softmax()

        # Khởi tạo trọng số ngẫu nhiên
        #self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.random_(0, 1)
            if module.bias is not None:
                module.bias.data.zero_()


    def forward(self, x):
        x = x.float()
        #print(x)
        x = self.activation_relu1(self.layer_hidden1(x))
        #print(x)
        x = self.activation_relu2(self.layer_hidden2(x))
        #print(x)
        #x = self.activation_relu3(self.layer_hidden3(x))
        x = self.activation_softmax(self.layer_output(x))
        #print(x)
        return x

# Hàm kích hoạt sigmoid
# def sigmoid(x):
#     return 1 / (1 + np.exp(-x))

# # Số lượng đầu vào, đầu ra và đơn vị ẩn
# input_size = 4
# hidden_size = 6
# output_size = 3

# # Khởi tạo trọng số ngẫu nhiên
# input_layer_weights = np.random.rand(input_size, hidden_size) #4x6
# hidden_layer_weights = np.random.rand(hidden_size, output_size) #6x3

# # Khởi tạo bias ngẫu nhiên
# input_layer_bias = np.random.rand(hidden_size)
# hidden_layer_bias = np.random.rand(output_size)

# def sigmoid_derivative(x):
#     # Đạo hàm của sigmoid
#     return x * (1 - x)

# #gradient descent: x+1 = x - learning_rate * gradient
# def backpropagation(input_data, target, output):
#     global input_layer_weights, hidden_layer_weights, input_layer_bias, hidden_layer_bias
    
#     # Tính sai số
#     output_error = target - output

#     # Tính gradient của đầu ra và đầu ra ẩn
#     output_delta = output_error * sigmoid_derivative(output)
#     hidden_layer_error = output_delta.dot(hidden_layer_weights.T)
#     hidden_layer_delta = hidden_layer_error * sigmoid_derivative(hidden_output)

#     # Cập nhật trọng số và bias
#     hidden_layer_weights += hidden_output.T.dot(output_delta)
#     hidden_layer_bias += np.sum(output_delta, axis=0, keepdims=True)
#     input_layer_weights += input_data.reshape(-1, 1).dot(hidden_layer_delta.reshape(1, -1))
#     input_layer_bias += np.sum(hidden_layer_delta, axis=0)

# # Hàm feedforward
# def feedforward(input_data):
#     # Tính đầu ra của lớp ẩn
#     hidden_input = np.dot(input_data, input_layer_weights) + input_layer_bias
#     hidden_output = sigmoid(hidden_input)
    
#     # Tính đầu ra cuối cùng
#     output = np.dot(hidden_output, hidden_layer_weights) + hidden_layer_bias
    
#     return output

# # Một ví dụ dữ liệu đầu vào
# input_data = np.array([0.2, 0.4, 0.6, 0.8])
# output_data = np.array([0.1, 0.3, 0.5])
# for epoch in range(5):
#     # Feedforward
#     output = feedforward(input_data)
#     print('Output: ', output)

#     # Backpropagation
#     backpropagation(input_data, output_data, output)