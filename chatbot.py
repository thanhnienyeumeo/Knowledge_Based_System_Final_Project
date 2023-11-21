import torch

from underthesea import word_tokenize
import numpy as np
import random
import json
from model import NeuralNetwork1, NeuralNetwork2

train_x = np.load('Data/train_x.npy')
train_y = np.load('Data/train_y.npy')
words = np.load('Data/words.npy')
classes = np.load('Data/classes.npy')
entity_list = [line.rstrip('\n') for line in open('Data/entity.dat')]
print(words)
model = NeuralNetwork1(len(train_x[0]), len(train_y[0]))

model.load_state_dict(torch.load('model.pth'))

def bag_of_words(s_words, words):
    bag = [0 for _ in range(len(words))]
    
    s_words = [word for word in s_words]
    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
    return np.array(bag)

def predict_class(tokenized, model, entity = None):
    # Filter out predictions below a threshold
    p = bag_of_words(tokenized, words)
    print("p: ", p)
    p = torch.from_numpy(p)
    res = model(p)
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    # Sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({ 'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list

name = None
choosed_part = None
last_intent = None
last_entity = None
json_data = None
data = None

while True:
    if not name:
        inp = input("Bot: Chào bạn! Tôi rất vui được giúp bạn. Phiền bạn có thể cho tôi tên của bạn?")
        name = inp
        continue
    if not choosed_part:
        inp = input('''Bot: Chào bạn {}. Bạn muốn hỏi về phần nào? Xin hãy nhập số tương ứng với chủ đề bạn muốn tôi trả lời
                    \n1.Luật chơi và các thông tin cơ bản về tennis
                    \n2.Thông tin về các tuyển thủ tennis nổi tiếng
                    \n3.Thông tin về các giải đấu tennis
                    \n4.Các tin tức mới nhất liên quan đến bộ môn tennis
                    \n5.Cách đánh tennis tốt hơn cùng các chiến thuật cơ bản trong tennis'''.format(name))
        #validate
        if inp not in ['1','2','3','4','5']:
            print("Bot: Xin lỗi, Bạn vui lòng hãy nhập số tương ứng với chủ đề bạn muốn tôi trả lời, từ 1 đến 5")
            continue
        choosed_part = int(inp)
        json_data = open("Data/intents_{}.json".format(choosed_part), encoding= 'utf-8').read()
        data = json.loads(json_data)
        print("Bot: Bạn muốn hỏi về phần {} phải không?".format(data['tag']))
        continue

    inp = input("You: ")
    if inp.lower() == "quit":
        if choosed_part is None:
            print("Bot: Tạm biệt {}. Nếu bạn cần các thông tin về tennis hãy quay lại với hệ thống dựa trên tri thức của chúng tôi.".format(name))
            break
        else:
            choosed_part = None
            json_data.close()
            continue

    inp = inp.lower()
    #find entity and intent
    s_words = word_tokenize(inp)
    for i in entity_list:
        if i.lower() in inp:
            last_entity = i
            break

    
    results = predict_class(s_words, model, last_entity)
    print(results)
    # last_intent = [intent for intent in data['intents'] if intent['tag'] == results[0]['intent']][0]
    # print(last_intent['responses'])