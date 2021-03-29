import matplotlib.pyplot as plt
import numpy as np


def feval(funcName, *args):  # function Handle
    return eval(funcName)(*args)


def RK4thOrder(func, yinit, x_range, h):  # fourth order method
    m = len(yinit)
    n = int((x_range[-1] - x_range[0])/h)

    x = x_range[0]
    y = yinit

    # Containers for solutions
    xsol = np.empty(0)
    xsol = np.append(xsol, x)  # time axis

    ysol = np.empty(0)
    ysol = np.append(ysol, y)  # array to store both x'(t) and y'(t)

    for i in range(n):  # Runge Kutta implimentation
        k1 = h*feval(func, x, y)

        yp2 = y + k1/2

        k2 = h*feval(func, x+h/2, yp2)

        yp3 = y + k2/2

        k3 = h*feval(func, x+h/2, yp3)

        yp4 = y + k3

        k4 = h*feval(func, x+h, yp4)

        for j in range(m):  # y = [x'(t_k), y'(t_k)]
            y[j] = y[j] + (1/6)*(k1[j] + 2*k2[j] + 2*k3[j] + k4[j])

        x = x + h  # updating time
        xsol = np.append(xsol, x)

        for r in range(len(y)):
            ysol = np.append(ysol, y[r])  # x'(t_k) and y'(t_k) appended

    return [xsol, ysol]


def myFunc(x, y):
    cD = 0.025
    m = 0.5  # mass
    k = cD/m
    g = 9.80665
    dy = np.zeros((len(y)))
    dy[0] = -k*np.sqrt((y[0])**2+(y[1])**2)*y[0]  # x'(t)
    dy[1] = -k*np.sqrt((y[0])**2+(y[1])**2)*y[1] - g  # y'(t)
    return dy

# -----------------------


h = 0.01  # stepsize
x = np.array([0.0, 40.0])  # range of x
yinit = np.array([25*np.sqrt(3), 25])  # y' IC
xpos = [0]
ypos = [0]

[ts, ys] = RK4thOrder('myFunc', yinit, x, h)
node = len(yinit)
ys1 = ys[0::node]  # splitting only x'(t_k)
ys2 = ys[1::node]  # splitting only y'(t_k)

for t in range(len(ts)):
    xpos.append(xpos[-1] + h*ys1[t])  # obtaining x(t_k) from x'(t_{k-1})
    ypos.append(ypos[-1] + h*ys2[t])  # obtaining y(t_k) from y'(t_{k-1})

for i in range(len(ypos)):
    if(ypos[i] < 0):
        print("Range", xpos[i-1])  # finding range
        print("Time of Flight", h*i)  # finding time of flight
        break

# ------plotting-----------
plt.plot(xpos, ypos, 'g')
plt.xlim(x[0], x[1])
plt.ylim(0, 15)
plt.xlabel('Horizontal distance', fontsize=12)
plt.ylabel('Vertical Distance', fontsize=12)
plt.tight_layout()
plt.show()
