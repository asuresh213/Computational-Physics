# Consider the boundary value problem y''=-pi^2*(y) with BC: y(0)=0, y(1/2)=1
# We know that the analytic solution is y = sin(pi*x)

import matplotlib.pyplot as plt
import numpy as np

# Definition of f_1 (x,y,z) = z. Please refer #4 in HW1 for context


def yprime(x, y, z):
    return z

# Definition of f_2(x,y,z) = z'. Please refer #4 in HW1 for context


def zprime(x, y, z):
    return -(np.pi)*(np.pi)*y


# -----------------initializing the variables---------------------------
x = []  # x value array to which we will add consecutive x values
y = []  # y value array that tracks the y-value for each x
z = []  # z = y' value array that tracks the y'-value for each x

xscaled = []
avalues = []
temp = []
graphdata = []
finalyvalues = []
goodvalues = []
x0 = 0
y0 = 0
y1 = 1
h = 0.01
endpt = 0.5
a = 2
iterations = int(endpt/h)
x.append(x0)
y.append(y0)
z.append(a)


while(a < 5):
    avalues.append(a)
    for i in range(1, iterations+1):
        y.append(y[-1] + yprime(x[-1], y[-1], z[-1])*h)
        z.append(z[-1] + zprime(x[-1], y[-1], z[-1])*h)
        x.append(x[-1]+h)
    finalyvalues.append(y[-1])
    if(a < 4.9):
        x = [0]
        y = [0]
        z = []
    a += 0.1
    z.append(a)

# plotting the actual sin(pi x) function
for i in x:
    xscaled.append(3.1415*i)

t = np.sin(xscaled)

# ---------------- building an array of 1s for plotting f(a)=1 ------------------
fun1 = []
for i in range(len(finalyvalues)):
    fun1.append(1)
# -------------------------------------------------------------------------------

plt.plot(avalues, finalyvalues)
plt.plot(avalues, fun1)
plt.vlines(np.pi, 0, finalyvalues[11], linestyle="dashed")
plt.xlabel("a")
plt.ylabel("y(1/2)")
plt.show()


for i in range(len(finalyvalues)):
    if(abs(finalyvalues[i]-fun1[i]) < 0.02):
        goodvalues.append(avalues[i])


# Plotting the approximations alongside the sin functions
for k in range(len(goodvalues)):
    x = [0]
    y = [0]
    z = [goodvalues[k]]
    for i in range(1, iterations+1):
        y.append(y[-1] + yprime(x[-1], y[-1], z[-1])*h)
        z.append(z[-1] + zprime(x[-1], y[-1], z[-1])*h)
        x.append(x[-1]+h)
    graphdata.append(y)
    z = []


plt.plot(x, graphdata[0], label="approx1")
plt.plot(x, graphdata[1], label="approx2")
plt.plot(x, t, label="sin(x)")
plt.legend()
plt.show()
