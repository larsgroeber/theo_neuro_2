import numpy as np
import matplotlib.pyplot as plt

N = 100
P_default = 5
t_max = 30

V = np.random.choice([-1, 1], (30, N))


def getM(v_1: np.array = None, P=P_default) -> np.array:
    if v_1 is not None:
        V[0] = v_1

    M = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            for p in range(P):
                M[i][j] += V[p][i] * V[p][j]

    np.fill_diagonal(M, 0)

    return M


def step(v: np.array, M: np.array):
    v_ = M.dot(v)
    v_ = [1 if i >= 0 else -1 for i in v_]
    return np.array(v_)


def overlap(v, v_m):
    return v.dot(v_m) / N


def random_v():
    M = getM()
    v = np.array([1 for i in range(N)])

    V = []

    for t in range(t_max):
        V.append(v)
        v = step(v, M)

    plt.matshow(V)
    plt.show()


def q_t(P, v_1):

    M = getM(v_1, P)
    v = np.array([1 for i in range(N)])
    qs = []
    print(f'initial overlap for P={P}: {overlap(v, v_1)}')

    for t in range(t_max):
        qs.append(overlap(v, v_1))
        v = step(v, M)

    print(f'final overlap for P={P}:   {qs[-1]}\n')
    return qs


# random_v()

change_elements = N/2
v_1 = np.array([1 for i in range(N)])
for i in np.random.randint(0, N, int(change_elements)):
    v_1[i] = -1

Ps = [i + 1 for i in range(30)]

Qs = [(q_t(p, v_1), p) for p in Ps]
for q in Qs:

    plt.plot(np.arange(0, t_max), q[0], label=f'P={q[1]}')

plt.xlabel('t')
plt.ylabel('q')
plt.legend()
plt.show()
