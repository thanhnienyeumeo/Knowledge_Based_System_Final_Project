import numpy as np
from underthesea import word_tokenize
# words = np.load("Data/Data_1/words.npy")
# print(words)
for i in range(1,5):
    prefix = 'Data/Data_{}'.format(i)
    test_x = np.array([])
    test_y = np.array([])
    try:
        classes = np.load(prefix + "/classes.npy")
        train_x = np.load(prefix + '/train_x.npy')
        train_y = np.load(prefix + '/train_y.npy')
        if(i == 1 or i == 5):
            test_x = np.load(prefix + '/test_x.npy')
            test_y = np.load(prefix + '/test_y.npy')
    except:
        continue
    for e,v in enumerate(classes):
        print(e,v)
    print('----- Chu de {} has {} classes and {} data-----'.format(i,len(classes),len(train_x)+len(test_x)))
   # print(len(train_x), len(test_x), len(train_y), len(test_y))
    
# A = np.load("train_x.npy")
# B = np.load("train_y.npy")
# for id,i in enumerate(A):
#     #print sentences index by words
#     for e,v in enumerate(i):
#         if v == 1:
#             print(words[e],end=' ')
#     for e,i in enumerate(B[id]):
#         if i == 1:
#             print(classes[e])
# entity_list = [line.rstrip('\n') for line in open('entity.dat')]
# print(entity_list)