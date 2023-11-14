import torch

from test import NeuralNetwork
from underthesea import word_tokenize
import numpy as np
import random
import json

train_x = np.load('train_x.npy')
train_y = np.load('train_y.npy')
words = np.load('words.npy')
classes = np.load('classes.npy')

model = NeuralNetwork(len(train_x[0]), len(train_y[0]))

model.load_state_dict(torch.load('model.pth'))

def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]
    s_words = word_tokenize(s)
    s_words = [word for word in s_words]
    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
    return np.array(bag)

def predict_class(s, model):
    # Filter out predictions below a threshold
    p = bag_of_words(s, words)
    p = torch.from_numpy(p)
    res = model(p)
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    # Sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list

# while True:
#     inp = input("You: ")
#     if inp.lower() == "quit":
#         break
#     results = predict_class(inp, model)
#     print(results)

inp = "Cho tôi biết một ít về luật tennis"
results = predict_class(inp, model)
print(results)