#%% 
import math
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
from graphviz import Digraph

#%%
# helper functions to draw the graph

def trace(root):
    nodes, edges = set(), set()
    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v._prev:
                edges.add((child, v))
                build(child)
    build(root)
    return nodes, edges

def draw_dot(root):
    dot = Digraph(format='svg', graph_attr={'rankdir': 'LR'})
    nodes, edges = trace(root)
    for n in nodes:
        uid = str(id(n))
        dot.node(name = uid, label= "{%s | data %.4f | grad %.4f}" % (n.label, n.data, n.grad, ), shape='record')
    
        if n._op:
            dot.node(name = uid + n._op, label = n._op)
            dot.edge(uid+ n._op, uid)

    for n1, n2 in edges:
        dot.edge(str(id(n1)), str(id(n2)) + n2._op)
    return dot

#%%
class Value:
    def __init__(self, data, _children=(), _op='', label = ""):
        self.data = data
        self._prev = _children
        self._op = _op
        self.label = label
        self.grad = 0.0 # we assume that initial value of the gradient is zero
        self._backward = lambda: None  # this will be defined later in the backward pass

    def __repr__(self):
         return f"Value(data={self.data})"
    
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)  # if other is not a Value, convert it to one
        out = Value(self.data + other.data, (self, other), "+")
        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)  # if other is not a Value, convert it to one
        out = Value(self.data * other.data, (self, other), "*")
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out
    
    def __rmul__(self, other):
        return self * other
    
    def __radd__(self, other):
        return self + other
    
    def tanh(self):
        out = Value(math.tanh(self.data), (self,), "tanh")
        def _backward():
            self.grad += (1 - math.tanh(self.data) ** 2) * out.grad
        out._backward = _backward
        return out
    
    def exp(self):
        out = Value(math.exp(self.data), (self,), "exp")
        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward
        return out
    
    def __truediv__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return self * other**-1
    
    def __neg__(self):
        return self * -1
    
    def __sub__(self, other):
        return self + (-other if isinstance(other, Value) else -Value(other))
    
    def __pow__(self, other):
        assert isinstance(other, (int, float)), "Power must be an integer or float"
        out = Value(self.data ** other, (self,), f"**{other}")
        def _backward():
            self.grad += (other * self.data ** (other - 1)) * out.grad
        out._backward = _backward
        return out
      
    def backward(self):
        # we will implement the backward pass later
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        self.grad = 1.0  # we set the gradient of the output to 1.0
        for v in reversed(topo):
            v._backward()
# %%
class Neuron:
    def __init__(self, nin):
        self.w = [Value(np.random.randn(), label=f'w{i}') for i in range(nin)]
        self.b = Value(np.random.randn(), label='b')

    def __call__(self, x):
        assert len(x) == len(self.w), "Input size must match number of weights"
        # calculate the weighted sum
        s = sum(w * x_i for w, x_i in zip(self.w, x)) + self.b
        return s.tanh()
    
    def parameters(self):
        return self.w + [self.b]

class Layer:
    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs  # return single output if only one neuron, otherwise return list of outputs

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]
    
class MLP:
    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(sz) - 1)]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
    
# %%
x = [Value(2.0, label='x1'), Value(3.0, label='x2'), Value(4.0, label='x3')]
n = MLP(3, [4,4,1]) # 3 inputs, 4 neurons in first layer, 4 in second, and 1 output neuron
o = n(x)
o.backward()
draw_dot(o)

# %% simulating a neuron
x1 = Value(2.0, label ='x1')
x2 = Value(0.0, label ='x2')
# weights
w1 = Value(-3.0, label ='w1')
w2 = Value(1.0, label ='w2')
b = Value(6.8813735870195432, label ='b')
x1w1 = x1 * w1; x1w1.label = "x1w1"
x2w2 = x2 * w2; x2w2.label = "x2w2"
x1w1x2w2 = x1w1 + x2w2; x1w1x2w2.label = "x1w1 + x2w2"
n = x1w1x2w2 + b; n.label = "n"
o = n.tanh(); o.label = "o"
o.backward()
draw_dot(o)

# %%
x1 = Value(2.0, label ='x1')
x2 = Value(0.0, label ='x2')
# weights
w1 = Value(-3.0, label ='w1')
w2 = Value(1.0, label ='w2')
b = Value(6.8813735870195432, label ='b')
x1w1 = x1 * w1; x1w1.label = "x1w1"
x2w2 = x2 * w2; x2w2.label = "x2w2"
x1w1x2w2 = x1w1 + x2w2; x1w1x2w2.label = "x1w1 + x2w2"
n = x1w1x2w2 + b; n.label = "n"
e = (2*n).exp()
o = (e-1) / (e+1)
o.label = "o"
o.backward()
draw_dot(o)
# %%
n = Value(5.0, label='n')
o = n.exp()
draw_dot(o)
# %%
xs = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0]
    ]
ys = [1.0, -1.0, -1.0, 1.0]

n = MLP(3, [4,4,1]) # 3 inputs, 4 neurons in first layer, 4 in second, and 1 output neuron

# %%
for k in range(20000):

    ypred = [n(x) for x in xs]
    #ypred
    loss = sum((yout - ygt)**2 for ygt, yout in zip(ys, ypred))
    for p in n.parameters():
        p.grad = 0.0

    loss.backward()
    for p in n.parameters():
        p.data += -0.1 * p.grad  # gradient descent step
print(k, loss.data)

ypred
# %%
ypred

# %%
