#%% 
import math
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

#%%

def f(x):
    return 3*x**2 - 4*x + 5

f(3.0)

#%%
xs = np.arange(-5,  5, 0.25)
ys = f(xs)
ys

#%%
plt.plot(xs, ys)

#%%
# calculate the derivative of f at x = 3.0
h = 1e-9
x = 3.0
(f(x+h) - f(x)) / h

#%%
# now will write a simple function
h = 0.001

a = 2.0
b = -3.0
c = 10
d1 = a*b + c
c += h
d2 = a*b + c
print("d1 = ", d1)
print("d2 = ", d2)
print("slope = ", (d2 - d1) / h)

#%%
# helper functions to draw the graph
from graphviz import Digraph

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
        dot.node(name = uid, label= "{%s | data %.4f}" % (n.label, n.data, ), shape='record')
    
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
    
    def __repr__(self):
         return f"Value(data={self.data})"
    
    def __add__(self, other):
        return Value(self.data + other.data, (self, other), "+")

    def __mul__(self, other):
        return Value(self.data * other.data, (self, other), "*")
    

a = Value(2.0, label = "a")
b = Value(-3.0, label = "b")
c = Value(10.0, label = "c")
e = b * a; e.label = "e"
d = e + c; d.label = "d"
f = Value(-2.0, label = "f")
L = d * f; L.label = "L"
draw_dot(L)

# %%
