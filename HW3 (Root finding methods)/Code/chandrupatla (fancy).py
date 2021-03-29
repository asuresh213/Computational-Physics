import numpy as np
import timeit
# ----------------------- Data --------------------------------------------------
Q = 3.1  # Volume rate of flow
b = 1.01  # width of the channel
h_0 = 0.83  # upstream water level
H = 0.1  # Height of the bump
g = -9.81  # Acceleration due to gravity

# ----------------------- Calculating the LHS -----------------------------------

alpha = ((Q**2)/(2*g*(b**2)*(h_0**2))) + h_0 - H


def chandrupatla(f, x0, x1):

    # Initialization
    b = x0
    a = x1
    c = x1
    fa = f(a)
    fb = f(b)
    fc = fa
    eps_m = None
    eps_a = None

    t = 0.5

    # jms: some guesses for default values of the eps_m and eps_a settings
    # based on machine precision... not sure exactly what to do here
    eps = np.finfo(float).eps
    if eps_m is None:
        eps_m = eps
    if eps_a is None:
        eps_a = 2*eps

    start = timeit.timeit()
    while True:
        # use t to linearly interpolate between a and b,
        # and evaluate this function as our newest estimate xt
        xt = a + t*(b-a)
        ft = f(xt)

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

        print (xt, ft)
    end = timeit.timeit()
    print("time", end-start)

# ----------------------- Setting up f(h) = 0 -----------------------------------


def func(h):
    return ((Q**2)/(2*g*(b**2)*(h**2)))+h - alpha


chandrupatla(func, 0.6, 0.8)  # calling the function
