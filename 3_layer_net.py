import sigmoid as sig
import numpy as np
def init_network():
    network = {}
    network['W1'] = np.array([[.1,.3,.5],[.2,.4,.6]])
    network['b1'] = np.array([.1,.2,.3])
    network['W2'] = np.array([[.1,.4],[.2,.5],[.3,.6]])
    network['b2'] = np.array([.1,.2])
    network['W3'] = np.array([[.1,.3],[.2,.4]])
    network['b3'] = np.array([.1,.2])
    return network

def forward(network,x):
    W1,W2,W3 = network['W1'],network['W2'],network['W3']
    b1,b2,b3 = network['b1'],network['b2'],network['b3']

    a1=np.dot(x,W1)+b1
    z1=sig.sigmoid(a1)
    a2=np.dot(z1,W2)+b2
    z2=sig.sigmoid(a2)
    a3=np.dot(z2,W3)+b2

    return a3

network=init_network()
x=np.array([1.0,.5])
y=forward(network,x)
print(y)
