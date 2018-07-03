import numpy as np
import matplotlib.pyplot as plt

sigma = 1.0
beta = 0.9
e = 0.01
cortex_size = 12.8
steps = 200
stimulus_per_step = 50

def delta_y(y, x, j):
  def W_j(x, y_j):
    return np.exp(-np.linalg.norm(x - y_j) ** 2 / (2 * sigma ** 2))

  def w_j(x_i, y, j):
    return W_j(x_i, y[j]) / sum(W_j(x_i, y_i) for y_i in y)

  def hebbian_j(x, y, j):
    return sum(w_j(x_i, y, j) for x_i in x) / len(x)

  def smooth_j(y, j):
    l = len(y)
    return y[(j-1) % l] - y[j] + y[(j+1) % l] - y[j]

  hebbian = e * hebbian_j(x, y, j)
  smooth = e * beta * smooth_j(y, j)
  return hebbian + smooth

shape = (20, 2)
y = np.zeros(shape)

for i, y_i in enumerate(y):
  y_i[0] = i * cortex_size / len(y)


def stimulus():
  return [
    np.random.uniform(high=cortex_size),
    np.random.normal(loc=0.0, scale=0.3)
  ]

st = [stimulus() for i in range(stimulus_per_step)]
for t in range(steps):
  print(f'{t/steps * 100}%')
  for i in range(len(y)):
    y[i] += delta_y(y, np.array(st), i)

f1 = [z[0] for z in y]
f2 = [z[1] for z in y]
s1 = [z[0] for z in st]
s2 = [z[1] for z in st]

plt.plot(f1, f2)
plt.plot(s1, s2, 'o')
plt.show()