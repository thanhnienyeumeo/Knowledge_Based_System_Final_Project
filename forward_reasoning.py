import json
import numpy as np

isTrue = np.zeros((5, 100))
#create a rules of reasoning
rules = [
    #0,0 -> 0,2; 0,11
    ([(0,10), (0, 13)] , [(0,7)]),
    ([(0,0)], [(0,2), (0,11), (0,12)]),
    ([(0,1)], [(0,5), (0,19)]),
    ([(0,15)], [(0,22)]),
    ([(0,15)], [(0,22)])
]
fact = { (0,0) }
#print([isTrue[x] for x in rules[0][1]])
def forward_reasoning(isTrue, fact, rules):
    all_new_fact = set()
    while True:
        is_new_fact = False
        new_fact = set()
        for r in rules:
            left = r[0]
            right = r[1]
            if all([isTrue[x] for x in right]):
                continue
            if all([x in fact for x in left]):
                for x in right:
                    if x not in fact:
                        is_new_fact = True
                        new_fact.add(x)
                        print('---new fact spotted: {} --- \n rule {} ---> rule {}'.format(x, left, right))
        if not is_new_fact:
            break
        fact = fact.union(new_fact)
        all_new_fact = all_new_fact.union(new_fact)
    return fact, all_new_fact
