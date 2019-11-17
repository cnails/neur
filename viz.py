from math import exp
import matplotlib.pyplot as plt
import numpy as np

x = np.arange(-15,15,0.1)

def func(x):
    return (1/(1+exp(-x))) * (1 - (1/(1+exp(-x))))

y = [func(i) for i in x]
plt.plot(x, y)
plt.show()
print(max(y))
