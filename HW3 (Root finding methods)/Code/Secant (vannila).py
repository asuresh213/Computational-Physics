import numpy as np
import matplotlib.pyplot as plt
import timeit
# ----------------------- Data -------------------------------------------------
Q = 3.1  # Volume rate of flow
b = 1.01  # width of the channel
h_0 = 0.83  # upstream water level
H = 0.1  # Height of the bump
g = -9.81  # Acceleration due to gravity
counter = 0  # counter to keep track of iterations
# ----------------------- Calculating the LHS ----------------------------------

alpha = ((Q**2)/(2*g*(b**2)*(h_0**2))) + h_0 - H

# ----------------------- Setting up f(h) = 0 -----------------------------------


def func(h):
    return ((Q**2)/(2*g*(b**2)*(h**2)))+h - alpha

# ----------------------- Secant method --------------------------------------


def secant(f, x0, x1):
    xvalues = [x0, x1]
    fvalues = [f(x0), f(x1)]
    counter = 0
    tol = 0.0000000001
    start = timeit.timeit()
    while True:
        counter += 1
        xr = xvalues[-1]
        xrmin1 = xvalues[-2]
        # added a small tolerance value to the denominator so we dont get divide by zero errors
        xnew = xr - f(xr)*((xr - xrmin1)/(f(xr) - f(xrmin1) + tol))
        xvalues.append(xnew)
        fvalues.append(f(xnew))
        print(counter, xvalues[-1], f(xvalues[-1]))
        if(np.abs(f(xr)-f(xrmin1)) < tol):
            break
    end = timeit.timeit()
    print("time", end-start)
    return 0


x0 = -1
iss = -0.1  # initial step size
x1 = x0 - func(x0)*((2*iss)/(func(x0+iss)-func(x0-iss)))
secant(func, x0, x1)
