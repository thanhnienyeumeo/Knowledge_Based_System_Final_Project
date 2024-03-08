import torch
from underthesea import word_tokenize
import numpy as np
import random
import json
from model import NeuralNetwork1, NeuralNetwork2, NeuralNetwork3
from forward_reasoning import forward_reasoning
from rules import rules



name = None
choosed_part = None
last_intent = None
json_data = None
data = None
fact = set() #
model = None
classes = None

def tokenize(sentence):
    s_words =  word_tokenize(sentence)
    for i in s_words:
        if i == "quần vợt":
            i = "tennis"
    return s_words

def bag_of_words(s_words, words):
    bag = [0 for _ in range(len(words))]
    
    s_words = [word for word in s_words]
    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
    return np.array(bag)

def predict_class(tokenized, model):
    # Filter out predictions below a threshold
    p = bag_of_words(tokenized, words)
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

def welcome_question():
    return input("Bot: Chào bạn! Tôi rất vui được giúp bạn. Phiền bạn có thể cho tôi tên của bạn?\nYou:")

def choose_part():
        inp = input('''Bot: Chào bạn {}. Bạn muốn hỏi về phần nào? Xin hãy nhập số tương ứng với chủ đề bạn muốn tôi trả lời
                    \n1.Luật chơi và các thông tin cơ bản về tennis
                    \n2.Thông tin về các tuyển thủ tennis nổi tiếng
                    \n3.Thông tin về các giải đấu tennis
                    \n4.Cách đánh tennis tốt hơn cùng các chiến thuật cơ bản trong tennis\nYou: '''.format(name))
        #validate
        if inp not in ['1','2','3','4', 'quit']:
            print("Bot: Xin lỗi, Bạn vui lòng hãy nhập số tương ứng với chủ đề bạn muốn tôi trả lời, từ 1 đến 4\n")
            return None, None, None, None, None, None
        if inp == "quit":
            return "quit", None, None, None, None, None
        choosed_part = int(inp)
        prefix = 'Data/Data_{}'.format(choosed_part)
        json_data = open(prefix + "/intents.json", encoding= 'utf-8').read()
        data = json.loads(json_data)
        


        words = np.load(prefix + '/words.npy')
        classes = np.load(prefix + '/classes.npy')
        train_x = np.load(prefix + '/train_x.npy')
        train_y = np.load(prefix + '/train_y.npy') #numpy file
        if choosed_part in range (1,5):
            model = NeuralNetwork1(len(train_x[0]), len(train_y[0]))
        else: 
            model = NeuralNetwork2(len(train_x[0]), len(train_y[0]))
        # load our saved model
        model.load_state_dict(torch.load(prefix + '/model.pth'))
        print("Bot: Tôi hiểu bạn muốn hỏi về phần {}. Hãy hỏi những câu hỏi bạn cần thắc mắc và tôi sẽ trả lời! Nếu bạn muốn thoát chủ đề, hãy nhập 'quit'".format(data['tag']))
        return choosed_part, model, data, words, classes, json_data

def reply(results, fact):
    global last_intent, choosed_part, model, data, words, classes, json_data
    if len(results) == 0:
        print("Bot: Xin lỗi, tôi không hiểu ý của bạn. Bạn có thể hỏi lại được không?\n")
        return

    last_intent = [intent for intent in data['intents'] if intent['tag'] == results[0]['intent']][0]
    answer = last_intent['responses'][0]
    print('Bot: ', answer)

    #suy dien
    index, = np.where(classes == last_intent['tag'])
    #print(index)
    new_data = (choosed_part - 1, index[0])
    fact.add(new_data)
    fact, new_fact = forward_reasoning(fact, rules)
    if len(new_fact) == 0: return fact

    goiY = "Có thể bạn muốn quan tâm: \n"
    for chuDe, chuDeCon in new_fact:
        prefix = 'Data/Data_{}'.format(chuDe + 1)
        recommend_class = np.load(prefix + "/classes.npy")
        if chuDe + 1 == choosed_part:
            goiY += recommend_class[chuDeCon] + "\n"
        else:
            goiY += recommend_class[chuDeCon] + "( chủ đề thứ " + str(chuDe + 1) + ")\n" 
    print(goiY)
    return fact

import warnings
warnings.filterwarnings("ignore")

while True:
    if not name:
        name = welcome_question()
        continue
    if not choosed_part:
        choosed_part, model, data, words, classes, json_data = choose_part()
        continue
    if choosed_part == "quit":
        print("Bot: Tạm biệt {}. Nếu bạn cần các thông tin về tennis hãy quay lại với hệ thống dựa trên tri thức của chúng tôi.\n".format(name))
        break
    inp = input("You: ")
    if inp.lower() == "quit":
            choosed_part = None
            continue
    s_words = tokenize(inp.lower())
    results = predict_class(s_words, model)
    fact = reply(results, fact)
    #print('Bot :' + random.choice(last_intent['responses']))