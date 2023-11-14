import random
import numpy as np
import json
import torch
import torch.nn as nn
import torch.optim as optim

class NeuralNetwork(nn.Module):
    def __init__(self, input_dimension, output_dimension):
        super(NeuralNetwork, self).__init__()
        
        # Lớp ẩn 1 với hàm kích hoạt ReLU
        self.layer_hidden1 = nn.Linear(input_dimension, 16)
        self.activation_relu1 = nn.ReLU()
        
        # Lớp ẩn 2 với hàm kích hoạt ReLU
        self.layer_hidden2 = nn.Linear(16, 32)
        self.activation_relu2 = nn.ReLU()
        
        # Lớp output với hàm kích hoạt softmax
        self.layer_output = nn.Linear(32, output_dimension)
        self.activation_softmax = nn.Softmax()

        # Khởi tạo trọng số ngẫu nhiên
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=1.0)
            if module.bias is not None:
                module.bias.data.zero_()


    def forward(self, x):
        x = x.float()
        #print(x)
        x = self.activation_relu1(self.layer_hidden1(x))
        #print(x)
        x = self.activation_relu2(self.layer_hidden2(x))
        #print(x)
        x = self.activation_softmax(self.layer_output(x))
        #print(x)
        return x

if __name__ == '__main__':
    with open('intents.json', encoding= 'utf-8') as json_data:
        intents = json.load(json_data)
        #print(intents)
        words = []
        classes = []
        documents = []
        stop_words = ['?', 'a', 'an', 'the']

        from underthesea import word_tokenize

        for intent in intents['intents']:
        
            for pattern in intent['patterns']:

                w = word_tokenize(pattern)
                words.extend(w)
                documents.append((w, intent['tag']))
                if intent['tag'] not in classes:
                    classes.append(intent['tag'])

        words = sorted(list(set(words)))

        #print(words)
        classes = sorted(list(set(classes)))
        #save words and classes
        np.save('words.npy', words)
        np.save('classes.npy', classes)
        
    #     training = []
    #     output = []
    #     output_empty = [0] * len(classes)

    #     # training set, bag of words for each sentence
    #     for doc in documents:
    #         bag = []
    #         pattern_words = doc[0]
    #         pattern_words = [word for word in pattern_words]

    #         for w in words:
    #             bag.append(1) if w in pattern_words else bag.append(0)

    #         # output is a '0' for each tag and '1' for current tag
    #         output_row = list(output_empty)
    #         output_row[classes.index(doc[1])] = 1

    #         training.append([bag, output_row])
    #     random.shuffle(training)
    # #  print(training)
    # #  print(len(training[0]), len(training[1]))
    #     training = np.array(training, dtype = object)

        
    #     # create train and test lists
    #     train_x = np.array(list(training[:,0]), dtype = float)
    #     train_y = np.array(list(training[:,1]), dtype = float)
    #     #save train_x and train_y 
    #     np.save('train_x.npy', train_x)
    #     np.save('train_y.npy', train_y)

    # # Build neural network
    #     # Define model and setup tensorboard

    #     model = NeuralNetwork(len(train_x[0]), len(train_y[0]))
    #     criterion = nn.CrossEntropyLoss()
    #     optimizer = optim.SGD(model.parameters(), lr=0.01)

    #     #Define data
    #     X = torch.from_numpy(train_x)
    #     Y = torch.from_numpy(train_y)
    #     train= torch.utils.data.TensorDataset(X,Y) 

    #     for epoch in range(10):
    #         cnt = 0
    #         for inputs, labels in train:
    #             # zero the parameter gradients
                
    #             #inputs = inputs.long()
    #             optimizer.zero_grad()
    #             # forward + backward + optimize
                
    #             outputs = model(inputs)
    #             outputs = outputs.unsqueeze(0)
    #             for i,v in enumerate(labels):
    #                 if v == 1:
    #                     print(outputs, i)
                
    #             labels = labels.unsqueeze(0)
                
    #         # print(labels.shape)
    #         #
    #             loss = criterion(outputs, labels)
    #             loss.backward()
    #             optimizer.step()
    #             print('Epoch: %d | Data: %d | Loss: %.4f' % (epoch+1, cnt, loss.item()))
    #             cnt+=1
    #     torch.save(model.state_dict(), 'model.pth')