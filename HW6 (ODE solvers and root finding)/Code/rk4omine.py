import numpy as np
import matplotlib.pyplot as plt


cD = 0.025
m = 0.5  # mass
k = cD/m
g = 9.80665
h = 0.001
counter = 0


def x2pr(xpr, ypr):
    x2pr = -k*np.sqrt((xpr)**2+(ypr)**2)*xpr  # x''(t)
    return x2pr


def y2pr(xpr, ypr):
    y2pr = -k*np.sqrt((xpr)**2+(ypr)**2)*ypr - g  # y''(t)
    return y2pr


def RK4o(xpr, ypr, h):
    k1x = h*x2pr(xpr, ypr)
    k1y = h*y2pr(xpr, ypr)
    xp2 = xpr + k1x/2
    yp2 = ypr + k1y/2
    k2x = h*x2pr(xp2, yp2)
    k2y = h*y2pr(xp2, yp2)
    xp3 = xpr + k2x/2
    yp3 = ypr + k2y/2
    k3x = h*x2pr(xp3, yp3)
    k3y = h*y2pr(xp3, yp3)
    xp4 = xpr + k3x
    yp4 = ypr + k3y
    k4x = h*x2pr(xp4, yp4)
    k4y = h*y2pr(xp4, yp4)
    return ([xpr + (1/6)*(k1x + 2*k2x + 2*k3x + k4x), ypr + (1/6)*(k1y + 2*k2y + 2*k3y + k4y)])


x = [0]
y = [0]
xpr = [25*np.sqrt(3)]
ypr = [25]

while counter < 10000000:
    x.append(x[-1] + h*xpr[-1])  # updating x positiion
    y.append(y[-1] + h*ypr[-1])  # updating y position
    xpr.append(RK4o(xpr[-1], ypr[-1], h)[0])
    ypr.append(RK4o(xpr[-2], ypr[-1], h)[1])
    if(y[-1] < 0):
        break

    counter += 1
print(x[-1])
plt.plot(x, y)
plt.xlim(0, 40)
plt.ylim(0, 10)
plt.show()
