import numpy as np
def NOT(x1):
    b=-.5
    tmp = x1+b
    if tmp >= 0:
        return 0
    else:
        return 1

def AND(x1,x2):
    x = np.array([x1,x2])
    w = np.array([.5,.5])
    b = -.7
    tmp = np.sum(w*x)+b
    if tmp <= 0:
        return 0
    else:
        return 1

def OR(x1,x2):
    x = np.array([x1,x2])
    w = np.array([.5,.5])
    b = -.3
    tmp = np.sum(w*x)+b
    if tmp <= 0:
        return 0
    else:
        return 1

def NAND(x1,x2):
    x = np.array([x1,x2])
    w = np.array([.5,.5])
    b = -.7
    tmp = np.sum(w*x)+b
    if tmp <= 0:
        return 1
    else:
        return 0

def XOR(x1,x2):
    c1 = AND(NOT(x1),x2)
    c2 = AND(x1,NOT(x2))
    return OR(c1,c2)
