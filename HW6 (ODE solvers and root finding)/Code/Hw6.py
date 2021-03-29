import numpy as np
import matplotlib.pyplot as plt

''' For forward euler time integration, we have
x[k+1] = x[k] + dt*x'[k]
y[k+1] = y[k] + dt*y'[k]
x'[k+1] = x'[k] + dt*x''[k]
y'[k+1] = y'[k] + dt*y''[k]
The range of the projectile would be the x[k] for k such that y[k] = 0
The time of flight would be the total length(x[])*dt '''

# ---------- Initialization ------------
cD = 0.025
m = 0.5  # mass
k = cD/m
g = 9.80665
x = [0]  # x-values
y = [0]  # y-values
xpr = [25*np.sqrt(3)]  # x' - values
ypr = [25]  # y'-values
dt = 0.001  # timestep

# ------- Double prime functions -------


def xdpr(xpr, ypr):
    return -1*k*np.sqrt(xpr**2 + ypr**2)*xpr


def ydpr(xpr, ypr):
    return -1*k*np.sqrt(xpr**2 + ypr**2)*ypr - g


while True:
    x.append(x[-1] + dt*xpr[-1])  # updating x positiion
    y.append(y[-1] + dt*ypr[-1])  # updating y position
    xpr.append(xpr[-1] + dt*xdpr(xpr[-1], ypr[-1]))  # updating the dx/dt
    ypr.append(ypr[-1] + dt*ydpr(xpr[-2], ypr[-1]))  # updating dy/dt
    if(y[-1] < 0):  # exit condition
        break


print("Range", x[-1])
print("Time of flight", len(x)*dt)
fig, ax = plt.subplots()
plt.plot(x, y)
ax.set_xlabel("horizontal distance")
ax.set_ylabel("vertical distance")
plt.show()
