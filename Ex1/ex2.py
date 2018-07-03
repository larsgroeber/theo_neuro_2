import numpy as np
import matplotlib.pyplot as plt


dt = 0.01
max_t = 10


class Hebbian:
    def __init__(self, w0, Q, tau, dt, max_t, alpha=0, w_bounds=None):
        self.w = w0
        self.Q = Q
        self.tau = tau
        self.dt = dt
        self.max_t = max_t
        self.alpha = alpha
        self.w_bounds = w_bounds
        self.w_t = []

    def _step(self):
        self.w += (1 / self.tau) * np.dot(self.Q, self.w) * self.dt - self.alpha * \
            np.dot(np.dot(np.dot(np.transpose(self.w), self.Q), self.w), self.w)
        if self.w_bounds != None:
            self. w = np.clip(self.w, self.w_bounds[0], self.w_bounds[1])

    def run(self):
        for t in range(int(self.max_t / self.dt)):
            self._step()
            self.w_t.append(np.copy(self.w))
        return self.w_t


Q = np.array([
    [1.0, -0.4],
    [-0.4, 1.0]
])


def test_run(w0, args=[], Q=Q):
    w0 = np.array(w0)

    h = Hebbian(w0, Q, 1.0, dt, max_t, *args)

    run = h.run()
    data = list(map(lambda x: np.linalg.norm(x), run))
    time = list(range(len(data)))
    return (time, data, run)


def a():
    init_cond = [
        [0.1, 0.1],
        [0.1, 1.0],
        [1.0, 1.0],
        [-1.0, -1.0]
    ]
    for index, w0 in enumerate(init_cond):
        data = test_run(w0)
        plt.subplot(2, 2, index + 1)
        plt.suptitle('a()')
        plt.title('w = {}'.format(w0))
        plt.ylabel('Length of w')
        plt.xlabel('Time')
        plt.plot(data[0], data[1], label='Length of w')
        plt.legend()

    plt.show()


def b():
    init_cond = [
        [0.5, 0.5],
        [0.6, 0.6],
        [0.7, 0.7],
    ]
    for index, w0 in enumerate(init_cond):
        data = test_run(w0, [0, (0, 1)])
        plt.subplot(2, 2, index + 1)
        plt.suptitle('b()')
        plt.title('w = {}'.format(w0))
        plt.ylabel('Length of w')
        plt.xlabel('Time')
        plt.plot(data[0], data[1], label='Length of w')
        plt.legend()

    plt.show()


def b2():
    Q2 = np.array([
        [1.0, -2.0],
        [-2.0, 1.0]
    ])
    init_cond = [
        [0.5, 0.5],
        [0.6, 0.6],
        [0.7, 0.7],
    ]
    for index, w0 in enumerate(init_cond):
        data = test_run(w0, Q=Q2)
        plt.subplot(2, 2, index + 1)
        plt.suptitle('b2()')
        plt.title('w = {}'.format(w0))
        plt.ylabel('Length of w')
        plt.xlabel('Time')
        plt.plot(data[0], data[1], label='Length of w')
        plt.plot(data[0], list(map(lambda l: l[0], data[2])), label='w1')
        plt.plot(data[0], list(map(lambda l: l[1], data[2])), label='w2')
        plt.legend()

    plt.show()


def c():
    init_cond = [
        [0.5, 0.5],
        [0.1, 0.1],
        [-0.5, -0.5],
        [-1.0, -1.0],
    ]
    for index, w0 in enumerate(init_cond):
        data = test_run(w0, [1])
        plt.subplot(2, 2, index + 1)
        plt.suptitle('c()')
        plt.title('w = {}'.format(w0))
        plt.ylabel('Length of w')
        plt.xlabel('Time')
        plt.plot(data[0], data[1], label='Length of w')
        plt.legend()

    plt.show()


a()
b()
b2()
c()
