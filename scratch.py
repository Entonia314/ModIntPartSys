import numpy as np

mass = [1,1,1]
g = 1
inits1 = np.array([[[-0.602885898116520, 1.059162128863347 - 1], [0.122913546623784, 0.747443868604908]],
                   [[0.252709795391000, 1.058254872224370 - 1], [-0.019325586404545, 1.369241993562101]],
                   [[-0.355389016941814, 1.038323764315145 - 1], [-0.103587960218793, -2.116685862168820]]])

p51 = 0.347111
p52 = 0.532728
inits3 = np.array([[[-1, 0], [p51, p52]],
                   [[1, 0], [p51, p52]],
                   [[0, 0], [-2 * p51, -2 * p52]]])

p61 = 0.464445
p62 = 0.396060
inits6 = np.array([[[-1, 0], [p61, p62]],
                   [[1, 0], [p61, p62]],
                   [[0, 0], [-2 * p61, -2 * p62]]])

def f(t, y):
    d0 = ((-g * mass[0] * mass[1] * (y[0] - y[1]) / np.linalg.norm(y[0] - y[1]) ** 3) +
          (-g * mass[0] * mass[2] * (y[0] - y[2]) / np.linalg.norm(y[0] - y[2]) ** 3)) / mass[0]
    d1 = ((-g * mass[1] * mass[2] * (y[1] - y[2]) / np.linalg.norm(y[1] - y[2]) ** 3) + (
            -g * mass[1] * mass[0] * (y[1] - y[0]) / np.linalg.norm(y[1] - y[0]) ** 3)) / mass[1]
    d2 = ((-g * mass[2] * mass[0] * (y[2] - y[0]) / np.linalg.norm(y[2] - y[0]) ** 3) + (
            -g * mass[2] * mass[1] * (y[2] - y[1]) / np.linalg.norm(y[2] - y[1]) ** 3)) / mass[2]
    return np.array([d0, d1, d2])


A = np.array([0, 2 / 9, 1 / 3, 3 / 4, 1, 5 / 6])
B = np.matrix([[0, 0, 0, 0, 0], [2 / 9, 0, 0, 0, 0], [1 / 12, 1 / 4, 0, 0, 0], [69 / 128, -243 / 128, 135 / 64, 0, 0],
               [-17 / 12, 27 / 4, -27 / 5, 16 / 15, 0], [65 / 432, -5 / 16, 13 / 16, 4 / 27, 5 / 144]])
C = np.array([1 / 9, 0, 9 / 20, 16 / 45, 1 / 12])
CH = np.array([47 / 450, 0, 12 / 25, 32 / 225, 1 / 30, 6 / 25])
CT = np.array([-1 / 150, 0, 3 / 100, -16 / 75, -1 / 20, 6 / 25])
eps = 0.0001


def runge_kutta_4(f, y0, t0, t1, h):
    """
    Explicit Euler method for systems of differential equations: y' = f(t, y); with f,y,y' n-dimensional vectors.
    :param f: list of functions
    :param y0: list of floats or ints, initial values y(t0)=y0
    :param t0: float or int, start of interval for parameter t
    :param t1: float or int, end of interval for parameter t
    :param h: float or int, step-size
    :return: two lists of floats, approximation of y at interval t0-t1 in step-size h and interval list
    """
    N = int(np.ceil((t1 - t0) / h))
    t = t0
    v = np.zeros((len(y0), N + 1, 2))
    y = np.zeros((len(y0), N + 1, 2))
    t_list = [0] * (N + 1)
    t_list[0] = t0
    v[:, 0, :] = y0[:, 1, :]
    y[:, 0, :] = y0[:, 0, :]
    for k in range(N):
        i = 0
        while i < (len(y0)):
            k1 = h * f(t + A[0] * h, y[:, k, :])[i]
            k2 = h * f(t + A[1] * h, (y[:, k, :] + B[1, 0] * h * k1))[i]
            k3 = h * f(t + A[2] * h, (y[:, k, :] + B[2, 0] * k1 + B[2, 1] * k2))[i]
            k4 = h * f(t + A[3] * h, (y[:, k, :] + B[3, 0] * k1 + B[3, 1] * k2 + B[3, 2] * k3))[i]
            k5 = h * f(t + A[4] * h, (y[:, k, :] + B[4, 0] * k1 + B[4, 1] * k2 + B[4, 2] * k3 + B[4, 3] * k4))[i]
            k6 = h * f(t + A[5] * h,
                       (y[:, k, :] + B[5, 0] * k1 + B[5, 1] * k2 + B[5, 2] * k3 + B[5, 3] * k4 + B[5, 4] * k5))[i]
            v[i, k + 1] = v[i, k] + CH[0] * k1 + CH[1] * k2 + CH[2] * k3 + CH[3] * k4 + CH[4] * k5 + CH[5] * k6

            l1 = v[i, k, :]
            l2 = v[i, k, :] + B[1, 0] * h * k1
            l3 = v[i, k, :] + B[2, 0] * k1 + B[2, 1] * k2
            l4 = v[i, k, :] + B[3, 0] * k1 + B[3, 1] * k2 + B[3, 2] * k3
            l5 = v[i, k, :] + B[4, 0] * k1 + B[4, 1] * k2 + B[4, 2] * k3 + B[4, 3] * k4
            l6 = v[i, k, :] + B[5, 0] * k1 + B[5, 1] * k2 + B[5, 2] * k3 + B[5, 3] * k4 + B[5, 4] * k5
            y[i, k + 1] = y[i, k] + h * CH[0] * l1 + h * CH[1] * l2 + h * CH[2] * l3 + h * CH[3] * l4 + h * CH[4] * l5 + h * CH[5] * l6

            TE = np.linalg.norm(CT[0] * l1 + CT[1] * l2 + CT[2] * l3 + CT[3] * l4 + CT[4] * l5 + CT[5] * l6)
            h_new = 0.9 * h * (eps/np.abs(TE)) ** (1/5)
            if abs(TE) > eps:
                h = h_new
            else:
                i = i+1

            i = i+1

        t = t + h
        t_list[k + 1] = t
    return y, t_list

def newton_raphson(f, g, x0, e, N):
    """
    Numerical solver of the equation f(x) = 0
    :param f: Function, left side of equation f(x) = 0 to solve
    :param g: Function, derivative of f
    :param x0: Float, initial guess
    :param e: Float, tolerable error
    :param N: Integer, maximal steps
    :return:
    """
    step = 1
    flag = 1
    condition = True
    while condition:
        if np.all(g(x0) == 0.0):
            print('Divide by zero error!')
            break
        x1 = x0 - f(x0) / g(x0)
        x0 = x1
        step = step + 1
        if step > N:
            flag = 0
            break
        condition = np.any(abs(f(x1)) > e)
    if flag == 1:
        return x1
    else:
        print('\nNot Convergent.')


def predictor_corrector(f, y0, t0, t1, N, ad_step):
    """
    Explicit Euler method for systems of differential equations: y' = f(t, y); with f,y,y' n-dimensional vectors.
    :param f: list of functions
    :param y0: list of floats or ints, initial values y(t0)=y0
    :param t0: float or int, start of interval for parameter t
    :param t1: float or int, end of interval for parameter t
    :param h: float or int, step-size
    :return: two lists of floats, approximation of y at interval t0-t1 in step-size h and interval list
    """
    h = (t1 - t0) / N
    v = np.zeros((len(y0), 1000000, 2))
    y = np.zeros((len(y0), 1000000, 2))
    t_list = [0] * (N + 2)
    t_list[0] = t0
    t = t0
    v[:, 0, :] = y0[:, 1, :]
    y[:, 0, :] = y0[:, 0, :]
    h_sum = 0
    h_min = h*0.1
    k = 1
    while h_sum < t1 and k < 50000:

        vp = v[:, k - 1, :] + h * f(t, y[:, k - 1, :])
        yp = y[:, k - 1, :] + h * v[:, k - 1, :] + h ** 2 * 0.5 * f(t, y[:, k-1, :])

        for i in range(len(y0)):

            def fixpoint(x):
                terms = []
                for j in range(len(x)):
                    if j != i:
                        term = (-g * mass[0] * mass[1] * (x[i, :] - x[j, :]) / np.linalg.norm(x[i, :] - x[j, :]) ** 3) / \
                               mass[i]
                        terms.append(term)
                return v[i, k - 1, :] + h * (terms[0] + terms[1]) - x[i, :]

            def fixpoint_deriv(x):
                terms = []
                for j in range(len(x)):
                    if j != i:
                        term = (g * mass[0] * mass[1] * (
                                2 * x[i, :] ** 2 - 3 * x[i, :] * x[j, :] + x[j, :] ** 2) / np.linalg.norm(
                            x[i, :] - x[j, :]) ** (5 / 2)) / mass[i]
                        terms.append(term)
                return h * (terms[0] + terms[1]) - 1

            vc = newton_raphson(fixpoint, fixpoint_deriv, yp, 0.001, 10)[i, :]

            v[i, k, :] = vc
            y[i, k, :] = y[i, k - 1, :] + h * v[i, k - 1, :] + h ** 2 * 0.5 * f(t, y[:, k - 1, :])[i]

        if ad_step == 1:
            err = np.linalg.norm(vp - v[:, k, :])

            if err > eps and h > h_min:
                h = h * 0.9
            elif err < eps ** 2:
                h_new = h * 1.1
                h = h_new
            else:
                k = k + 1
                h_sum += h
        else:
            k = k + 1
            h_sum += h

        t = t + h
        #t_list[k] = t
    y = y[:, :k + 1, :]
    return y, t_list


def forward_euler(f, y0, t0, t1, N):
    """
    Explicit Euler method for systems of differential equations: y' = f(t, y); with f,y,y' n-dimensional vectors.
    :param f: list of functions
    :param y0: list of floats or ints, initial values y(t0)=y0
    :param t0: float or int, start of interval for parameter t
    :param t1: float or int, end of interval for parameter t
    :param h: float or int, step-size
    :return: two lists of floats, approximation of y at interval t0-t1 in step-size h and interval list
    """
    h = (t1 - t0) / N
    h_min = h / 16
    h_sum = 0
    k = 0
    t = t0
    v = np.zeros((len(y0), 1000000, 2))
    y = np.zeros((len(y0), 1000000, 2))
    t_list = [0] * (N + 1)
    t_list[0] = t0
    v[:, 0, :] = y0[:, 1, :]
    y[:, 0, :] = y0[:, 0, :]
    e_max = np.linalg.norm(mass[0] * f(t, y[:, 0, :])[0] + mass[1] * f(t, y[:, 0, :])[1] + mass[2] * f(t, y[:, 0, :])[2])
    #print(e_max)
    while h_sum < t1 and k < 50000:
        for i in range(len(y0)):
            y[i, k + 1, :] = y[i, k, :] + h * v[i, k, :]
            v[i, k + 1, :] = v[i, k, :] + h * f(t, y[:, k, :])[i]
        energy = np.linalg.norm(mass[0] * f(t, y[:, k, :])[0] + mass[1] * f(t, y[:, k, :])[1] + mass[2] * f(t, y[:, k, :])[2])
        #print(energy)
        if abs(energy) < np.float64 * 2:
            k = k + 1
            h_sum = h_sum + h
            if abs(energy) == 0:
                h = h * 2
        elif h > h_min:
            h = h * 0.5
        else:
            k = k + 1
            h_sum = h_sum + h
            print('h zu klein')
        t = t + h
        #t_list[k + 1] = t
    y = y[:, :k + 1 , :]
    #print(y)
    print(k)
    #print(e_max)
    return y, t


y, t = forward_euler(f, inits3, 0, 10, 1000)
#print(y)

