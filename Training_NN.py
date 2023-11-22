import random
import numpy as np
import json
import torch
import torch.nn as nn
import torch.optim as optim

from model import NeuralNetwork1, NeuralNetwork2

if __name__ == "__main__":
    for i in range(5,6):
        prefix = 'Data/Data_{}'.format(i)
        train_x = np.load(prefix + '/train_x.npy')
        train_y = np.load(prefix + '/train_y.npy')
        isTest = 1
        
    # Build neural network
        # Define model and setup tensorboard
        model = None
        if(len(train_y[0]) <= 8):
             model = NeuralNetwork2(len(train_x[0]), len(train_y[0]))
             isTest = 0
        else:
            model = NeuralNetwork1(len(train_x[0]), len(train_y[0]))
        if isTest == 1:
            test_x = np.load(prefix + '/test_x.npy')
            test_y = np.load(prefix + '/test_y.npy')
        # Define loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.03)

        #Define data
        X = torch.from_numpy(train_x)
        Y = torch.from_numpy(train_y)
        train= torch.utils.data.TensorDataset(X,Y) 
        
        if isTest == 1:
            X = torch.from_numpy(test_x)
            Y = torch.from_numpy(test_y)
            test= torch.utils.data.TensorDataset(X,Y)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        num_epoch = 1000
        for epoch in range(num_epoch):
            sum_epoch = 0
            cnt = 0
            for inputs, labels in train:
                # zero the parameter gradients
                
                #inputs = inputs.long()
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                # forward + backward + optimize
                
                outputs = model(inputs)
                outputs = outputs.unsqueeze(0)
                if epoch == num_epoch - 1:
                    for i,v in enumerate(labels):
                        if v == 1:
                            print(outputs, i)
                
                labels = labels.unsqueeze(0)
                
            # print(labels.shape)
            #
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                #print('Epoch: %d | Data: %d | Loss: %.4f' % (epoch+1, cnt, loss.item()))
                cnt+=1
                sum_epoch += loss.item()
            print('Epoch: %d | Loss: %.4f' % (epoch+1, sum_epoch))
            #validate
            if isTest and (epoch+1) % 5 == 0:
                with torch.no_grad():
                    correct = 0
                    total = 0
                    for inputs, labels in test:
                        inputs = inputs.to(device)
                        labels = labels.to(device)
                        outputs = model(inputs)
                        
                        predicted = torch.argmax(outputs.data)
                       
                        
                        total += 1
                        #find the 1 on labels
                        correct += (predicted == torch.argmax(labels))
                        #.sum.item()?
                    print('Epoch: %d | Accuracy: %.4f %%' % (epoch+1, 100 * correct / total))
            #print()
        torch.save(model.state_dict(), prefix + '/model.pth')