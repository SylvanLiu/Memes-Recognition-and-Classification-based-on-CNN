import matplotlib.pyplot as plt
import numpy as np
import random
import matplotlib.tri as tri
count = 100
X = [random.random()*20 for i in range(count)]
Y = [random.random()*10 for i in range(count)]
triangles = tri.Triangulation(X, Y)
plt.triplot(triangles,'r--')
plt.show()
