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

class NeuralNetwork3(nn.Module):
    def __init__(self, input_dimension, output_dimension):
        super(NeuralNetwork3, self).__init__()
        
        # Lớp ẩn 1 với hàm kích hoạt ReLU
        self.layer_hidden1 = nn.Linear(input_dimension, 16)
        self.activation_relu1 = nn.ReLU()
        
        # Lớp ẩn 2 với hàm kích hoạt ReLU
        self.layer_hidden2 = nn.Linear(16, 32)
        self.activation_relu2 = nn.ReLU()

        # Lớp ẩn 3 với hàm kích hoạt ReLU
        self.layer_hidden3 = nn.Linear(32, 64)
        self.activation_relu3 = nn.ReLU()
        
        # Lớp output với hàm kích hoạt softmax
        self.layer_output = nn.Linear(64, output_dimension)
        self.activation_softmax = nn.Softmax()

        # Khởi tạo trọng số ngẫu nhiên
        #self.apply(self._init_weights)


    def forward(self, x):
        x = x.float()
        #print(x)
        x = self.activation_relu1(self.layer_hidden1(x))
        #print(x)
        x = self.activation_relu2(self.layer_hidden2(x))
        #print(x)
        x = self.activation_relu3(self.layer_hidden3(x))
        #x = self.activation_relu3(self.layer_hidden3(x))
        x = self.activation_softmax(self.layer_output(x))
        #print(x)
        return x