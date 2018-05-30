import numpy as np
def softmax(a):
    c=np.max(a)
    a1=np.exp(a-c)
    a2=np.sum(a1)
    return a1/a2
