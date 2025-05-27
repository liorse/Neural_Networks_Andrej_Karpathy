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


# %%
class Value:
    def __init__(self, data):
        self.data = data

    def __repr__(self):
         return f"Value(data={self.data})"
    
    def __add__(self, other):
        return Value(self.data + other.data)

    def __mul__(self, other):
        return Value(self.data * other.data)
    

a = Value(2.0)
b = Value(-3.0)
c = Value(10.0)
b*a + c