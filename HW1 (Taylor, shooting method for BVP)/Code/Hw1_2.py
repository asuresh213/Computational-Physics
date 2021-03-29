import matplotlib.pyplot as plt
import numpy as np

diffarray1 = []
diffarray2 = []
diffarray3 = []
tempsum = 0


def fact(n):
    temp = 1
    for i in range(1, n+1):
        temp *= i
    return temp


def p(x, nmax):
    tempsum = 0
    for n in range(nmax+1):
        nfact = fact(2*n)
        tempsum += (-1)**n * ((x**(2*n))/nfact)
    return tempsum


x = np.arange(0, 4*np.pi, 0.1)
y = np.cos(x)
z = p(x, 1)
w = p(x, 2)
t = p(x, 10)


plt.plot(x, y, label="cos(x)")
plt.plot(x, z, label="Leading order approximation")
plt.plot(x, w, label="Fourth order approximation")
plt.plot(x, t, label="Approximation upto 10 terms")
plt.axis([0, 4*np.pi, -2.5, 2.5])
plt.legend()
plt.show()


for r in range(len(x)):
    diffarray1.append(abs(y[r]-z[r]))
    diffarray2.append(abs(y[r]-w[r]))
    diffarray3.append(abs(y[r]-t[r]))

plt.plot(x, diffarray1, label="second order")
plt.plot(x, diffarray2, label="fourth order")
plt.plot(x, diffarray3, label="20th order")
plt.xlabel("x")
plt.ylabel("|p_n(x)-cos(x)|")
plt.legend()
plt.show()
