import numpy as np
from torch import Tensor
a = [1,2,3]
a = Tensor(a)
for i,v in enumerate(a):
    print(i,v)