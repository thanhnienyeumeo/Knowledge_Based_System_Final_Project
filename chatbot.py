import torch

from underthesea import word_tokenize
import numpy as np
import random
import json
from model import NeuralNetwork1, NeuralNetwork2
from forward_reasoning import isTrue, rules, forward_reasoning


entity_list = [line.rstrip('\n') for line in open('Data/entity.dat')]


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
fact = set()

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
        prefix = 'Data/Data_{}'.format(choosed_part)
        json_data = open(prefix + "/intents.json", encoding= 'utf-8').read()
        data = json.loads(json_data)
        


        words = np.load(prefix + '/words.npy')
        classes = np.load(prefix + '/classes.npy')
        train_x = np.load(prefix + '/train_x.npy')
        train_y = np.load(prefix + '/train_y.npy')
        if choosed_part in [1,4]:
            model = NeuralNetwork1(len(train_x[0]), len(train_y[0]))
        else: 
            model = NeuralNetwork2(len(train_x[0]), len(train_y[0]))
        # load our saved model
        model.load_state_dict(torch.load(prefix + '/model.pth'))
        print("Bot: Tôi hiểu bạn muốn hỏi về phần {}. Hãy hỏi những câu hỏi bạn cần thắc mắc và tôi sẽ trả lời!".format(data['tag']))
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
    print(results, last_entity)

    last_intent = [intent for intent in data['intents'] if intent['tag'] == results[0]['intent']][0]
    index, = np.where(classes == last_intent['tag'])

    if isTrue[(choosed_part, index)]: 
        print('ok')
        continue
    isTrue[(choosed_part, index)] = 1
    fact, new_fact = forward_reasoning(isTrue, fact, rules)