import numpy as np
from underthesea import word_tokenize
words = np.load("Data/Data_1/words.npy")
print(words)
classes = np.load("Data/Data_1/classes.npy")
print(classes)
print(len(words), len(classes))
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