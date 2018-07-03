import numpy as np
import matplotlib.pyplot as plt

NUM_CELLS = 512
SIZE_OF_CORTEX = NUM_CELLS
SIGMA = 0.066
a_matrix = np.linspace(0, SIZE_OF_CORTEX, NUM_CELLS)

MAX_T = 50
DT = 1


class Hebbian:
    def __init__(self, w0, Q, K, tau, dt, epsilon, max_t, alpha=0, w_bounds=None):
        self.w = w0
        self.Q = Q
        self.K = K
        self.epsilon = epsilon
        self.tau = tau
        self.dt = dt
        self.max_t = max_t
        self.t = []
        self.alpha = alpha
        self.w_bounds = w_bounds
        self.w_t = []

    def _step(self):
        self.w += self.w + self.epsilon * self.K.dot(self.w).dot(self.Q)
        if self.w_bounds != None:
            self. w = np.clip(self.w, self.w_bounds[0], self.w_bounds[1])

    def run(self):
        for t in range(int(self.max_t / self.dt)):
            self._step()
            self.w_t.append(np.copy(self.w))
            self.t.append(t)
        return self.w_t


def generate_K(a, a_):
    return np.exp(-(a - a_) ** 2/(2 * SIGMA ** 2)) - 1 / 9 * np.exp(-(a - a_) ** 2/(18 * SIGMA ** 2))


K = np.empty((NUM_CELLS, NUM_CELLS))

for i in range(NUM_CELLS):
    for j in range(NUM_CELLS):
        K[i][j] = generate_K(i, j) + generate_K(i + NUM_CELLS,
                                                j) + generate_K(i, j + NUM_CELLS)


w_R = np.random.random((NUM_CELLS)) / 10
w_L = np.random.random((NUM_CELLS)) / 10

# w_R = np.array([0.1 for i in range(NUM_CELLS)])
# w_L = np.array([0.1 for i in range(NUM_CELLS)])
# w_R[100] = 1
# w_R[101] = 1
# w_R[102] = 1

W = np.transpose(np.array([w_R, w_L]))

# plt.title('K')
# plt.matshow(K)
# plt.show()

Q = np.array([
    [1, -2],
    [-2, 1],
])

h = Hebbian(W, Q, K, 1, DT, 0.01, MAX_T, w_bounds=[0, 1])
result = h.run()

result_minus = list(map(lambda x: list(map(lambda y: y[0] - y[1], x)), result))
result_plus = list(map(lambda x: list(map(lambda y: y[0] + y[1], x)), result))

# plt.subplot(2, 1, 1)
# plt.title('w_r - w_l')
plt.matshow(result_minus)
# plt.subplot(2, 1, 2)
# plt.title('w_r + w_l')
plt.matshow(result_plus)

# for index, i in enumerate([1, 3, 5, 7]):

#     to_plot = result[i]

#     plt.subplot(2, 2, index + 1)
#     plt.plot(list(range(NUM_CELLS)), [r[0] - r[1]
#                                       for r in to_plot], label=f't={i}')


# plt.show()

# for index, i in enumerate([1, 3, 5, 7]):

#     to_plot = result[i]

#     plt.subplot(2, 2, index + 1)
#     plt.plot(list(range(NUM_CELLS)), [r[0] + r[1]
#                                       for r in to_plot], label=f't={i}')
plt.show()
