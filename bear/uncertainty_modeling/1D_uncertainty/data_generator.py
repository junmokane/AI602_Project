import numpy as np
import matplotlib.pyplot as plt

x = np.concatenate([np.linspace(-10, -6, 20), np.linspace(-2, 2, 20), np.linspace(6, 10, 20)])
noise = np.random.normal(0, 0.1, size=60)
y = (np.sin(x) + noise)

x = np.reshape(x,[-1,1])
y = np.reshape(y,[-1,1])
np.save('reg_data/data2', np.hstack([x, y]))
print(np.hstack([x,y]).shape)

plt.figure(figsize=(4,3))
plt.scatter(x, y)
plt.show()


'''
# x1 = np.repeat(np.linspace(1, 3 ,100), 5)
x1 = np.linspace(1, 3 ,100)
# x2 = np.repeat(np.linspace(-1, 1, 100), 10)
# x3 = np.repeat(np.linspace(5, 7, 100), 5)
x3 = np.linspace(5, 7, 100)

y1 = x1  + np.random.normal(0, 0.1, len(x1))
# y2 = np.sin(x2 * 2) + np.random.normal(0, 0.1, len(x2))
# y2 = np.random.normal(0, 10, len(x2))
# y2 = np.random.rand(len(x2)) * 8 - 4
y3 = x3 + np.random.normal(0, 0.1, len(x3))

# x = np.concatenate([x1, x2, x3])
# y = np.concatenate([y1, y2, y3])
x = np.concatenate([x1, x3])
y = np.concatenate([y1, y3])

x = np.repeat(np.linspace(-10, 10, 20), 20)
y = np.repeat(np.random.rand(20), 20) + np.random.normal(0, 0.01, 400)

x = np.append(np.linspace(-5, -2, 100), np.linspace(2, 5, 100))
'''