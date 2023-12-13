import random
import numpy as np
import json
import torch
import torch.nn as nn
import torch.optim as optim

from model import NeuralNetwork1, NeuralNetwork2, NeuralNetwork3

if __name__ == "__main__":
    for i in [2,3]:
        prefix = 'Data/Data_{}'.format(i)
        train_x = np.load(prefix + '/train_x.npy')
        train_y = np.load(prefix + '/train_y.npy')
        test_x = np.load(prefix + '/test_x.npy')
        test_y = np.load(prefix + '/test_y.npy')
        best_acc = 0
    # Build neural network
        # Define model and setup tensorboard
        for name_model in [1,2,3]:
            model = None
            if name_model == 1:
                model = NeuralNetwork1(len(train_x[0]), len(train_y[0]))
            elif name_model == 2:
                model = NeuralNetwork2(len(train_x[0]), len(train_y[0]))
            elif name_model == 3:
                model = NeuralNetwork3(len(train_x[0]), len(train_y[0]))
            
            
            # Define loss and optimizer
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.SGD(model.parameters(), lr=0.03)

            #Define data
            X = torch.from_numpy(train_x)
            Y = torch.from_numpy(train_y)
            train= torch.utils.data.TensorDataset(X,Y) 
            X = torch.from_numpy(test_x)
            Y = torch.from_numpy(test_y)
            test= torch.utils.data.TensorDataset(X,Y)

            device =  torch.device("cpu")
            # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            print("---- Start traing data {} by model {}----".format(i, name_model))

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
                    
                    
                    labels = labels.unsqueeze(0)
                    
                # print(labels.shape)
                #
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    #print('Epoch: %d | Data: %d | Loss: %.4f' % (epoch+1, cnt, loss.item()))
                    cnt+=1
                    sum_epoch += loss.item()
                
                #validate
                if epoch == num_epoch - 1:
                    print("final loss is: %d", sum_epoch)
                if (epoch+1) % 100 == 0:
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
                        print('Epoch: %d | Loss: %.4f' % (epoch+1, sum_epoch))
                        print('Epoch: %d | Accuracy: %.4f %%' % (epoch+1, 100 * correct / total))
                #print()
            print("---- Finish traing data {} in model {}, accuracy is {} ----".format(i, name_model, 100*correct/total))
            if correct/total > best_acc:
                best_acc = 100 * correct/total
                print("best acc is: %.4f" % (best_acc))
                torch.save(model.state_dict(), prefix + '/model.pth')