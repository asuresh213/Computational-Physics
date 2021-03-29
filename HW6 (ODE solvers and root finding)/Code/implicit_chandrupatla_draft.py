import numpy as np
import timeit
import matplotlib.pyplot as plt
# ----------------------- Data --------------------------------------------------
cD = 0.025
m = 0.5  # mass
k = cD/m
g = 9.80665
x = [0]  # x-values
y = [0]  # y-values
u = [25*np.sqrt(3)]  # x' - values
v = [25]  # y'-values
counter = 0  # ensuring a safer exit condition
dt = 0.01  # timestep
# ----------------------- Calculating the LHS -----------------------------------


def chandrupatla(f, curr_u, curr_v, x0, x1):

    # Initialization
    b = x0
    a = x1
    c = x1
    fa = f(a, curr_u, curr_v)
    fb = f(b, curr_u, curr_v)
    fc = fa
    eps_m = None
    eps_a = None

    t = 0.5

    # jms: some guesses for default values of the eps_m and eps_a settings
    # based on machine precision... not sure exactly what to do here
    #eps = np.finfo(float).eps
    eps = 2
    if eps_m is None:
        eps_m = eps
    if eps_a is None:
        eps_a = 2*eps

    start = timeit.timeit()
    while True:
        # use t to linearly interpolate between a and b,
        # and evaluate this function as our newest estimate xt
        xt = a + t*(b-a)
        ft = f(xt, curr_u, curr_v)

        if np.sign(ft) == np.sign(fa):
            c = a
            fc = fa
        else:
            c = b
            b = a
            fc = fb
            fb = fa
        a = xt
        fa = ft

        # set xm so that f(xm) is the minimum magnitude of f(a) and f(b)
        if np.abs(fa) < np.abs(fb):
            xm = a
            fm = fa
        else:
            xm = b
            fm = fb
        if fm == 0:
            break

        # Figure out values xi and phi
        # to determine which method we should use next
        tol = 2*eps_m*np.abs(xm) + eps_a
        tlim = tol/np.abs(b-c)
        if tlim > 0.5:
            break

        xi = (a-b)/(c-b)
        phi = (fa-fb)/(fc-fb)

        if(phi**2 < xi and (1-phi)**2 < 1-xi):
            # inverse quadratic interpolation
            t = fa / (fb-fa) * fc / (fb-fc) + (c-a)/(b-a)*fa/(fc-fa)*fb/(fc-fb)
        else:
            # bisection
            t = 0.5

        # limit to the range (tlim, 1-tlim)
        t = np.minimum(1-tlim, np.maximum(tlim, t))

    return xt

# ----------------------- Setting up f(h) = 0 -----------------------------------


def horizontal(h, curr_u, curr_v):
    return h - curr_u + dt*(k*np.sqrt(h**2 + ((h/curr_u)*(curr_v - (g*dt)))**2)*h)


def vertical(h, curr_u, curr_v):
    return h - curr_v + dt*(k*np.sqrt(h**2 + ((curr_u*h)/(curr_v - (g*dt)))**2)*h + g)


while counter < 100:
    u.append(chandrupatla(horizontal, u[-1], v[-1], 0, 1))  # calling the function
    x.append(x[-1] + dt*u[-1])
    v.append(chandrupatla(vertical, u[-2], v[-1], 1, 1000))  # calling the function
    print(v[-1])
    y.append(y[-1]+dt*v[-1])
    counter += 1

plt.plot(x, y)
plt.show()
