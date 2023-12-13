import numpy as np
import random
import json
from underthesea import word_tokenize

if __name__ == '__main__':
    for i in range(1,5):
        prefix = 'Data/Data_{}'.format(i)
        with open(prefix + '/intents.json', encoding= 'utf-8') as json_data:
            intents = json.load(json_data)
            print(intents)
            words = []
            classes = []
            documents = {}
            from underthesea import word_tokenize

            documents['train'] = []
            documents['test'] = []
            ratio = 0.8
            for intent in intents['intents']:
                print(intent['patterns'], type(intent['patterns']))
                lenn = len(intent['patterns'])
                for i,pattern in enumerate(intent['patterns']):
                    pattern = pattern.lower()
                    w = word_tokenize(pattern)
                    
                    words.extend(w)
                    if(i < ratio * lenn):
                        documents['train'].append((w, intent['tag']))
                    else:
                        documents['test'].append((w, intent['tag']))
                    if intent['tag'] not in classes:
                        classes.append(intent['tag'])

            words = sorted(list(set(words)))

            print(words, len(words))
            
            print(classes)
            #save words and classes
            np.save(prefix + '/words.npy', words)
            np.save(prefix + '/classes.npy', classes)
            
            training = []
            testing = []
            output = []
            output_empty = [0] * len(classes)

            # training set, bag of words for each sentence
            for doc in documents['train']:
                bag = []
                pattern_words = doc[0]
                pattern_words = [word for word in pattern_words]

                for w in words:
                    bag.append(1) if w in pattern_words else bag.append(0)

                # output is a '0' for each tag and '1' for current tag
                output_row = list(output_empty)
                output_row[classes.index(doc[1])] = 1

                training.append([bag, output_row])
            random.shuffle(training)
            for doc in documents['test']:
                bag = []
                pattern_words = doc[0]
                pattern_words = [word for word in pattern_words]

                for w in words:
                    bag.append(1) if w in pattern_words else bag.append(0)

                # output is a '0' for each chuDe and '1' for current chuDe
                output_row = list(output_empty)
                output_row[classes.index(doc[1])] = 1

                testing.append([bag, output_row])
            random.shuffle(testing)
        #  print(training)
        #  print(len(training[0]), len(training[1]))
            training = np.array(training, dtype = object)
            testing = np.array(testing, dtype = object)
            #split training to train and test, according to ratio 75:25
    
            # create train and test lists
            train_x = np.array(list(training[:,0]), dtype = float)
            train_y = np.array(list(training[:,1]), dtype = float)
            np.save(prefix + '/train_x.npy', train_x)
            np.save(prefix + '/train_y.npy', train_y)
            if testing.size == 0:
                continue
            test_x = np.array(list(testing[:,0]), dtype = float)
            test_y = np.array(list(testing[:,1]), dtype = float)

            #save train_x and train_y 
            
            np.save(prefix + '/test_x.npy', test_x)
            np.save(prefix + '/test_y.npy', test_y)