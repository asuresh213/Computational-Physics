import numpy as np
import matplotlib.pyplot as plt
import timeit

# -------------------------- Function definitons --------------------------------


def forwardeuler(f, x, h):
    return (f(x+h) - f(x))/h


def backwardeuler(f, x, h):
    return (f(x) - f(x-h))/h


def centralDifference(f, x, h):
    return (f(x+h) - f(x-h))/(2*h)


def g(x):
    return (np.cos(5*np.pi*x) - np.e**(-(x**2)/2)*np.sin(0.5*np.pi*x))*(np.sin(np.pi*x)/x)


def gpr(x):
    m = np.cos(5*np.pi*x) - np.e**(-(x**2)/2)*np.sin(0.5*np.pi*x)
    n = np.sin(np.pi*x)/x
    mpr = -5*np.pi*np.sin(5*np.pi*x) - (np.e**(-(x**2)/2)*(-x)*np.sin(0.5 *
                                                                      np.pi*x) + 0.5*np.pi*np.cos(0.5*np.pi*x)*np.e**(-(x**2)/2))
    npr = -(np.sin(np.pi*x)/x**2) + (1/x)*(np.pi*np.cos(np.pi*x))
    return (m*npr + n*mpr)


def gdpr(x):
    return -(np.pi**2*np.sin(np.pi*x)*(np.cos(5*np.pi*x) - np.e**(-x**2/2)*np.sin(1.5708*x)))/x + 2*np.pi*np.cos(np.pi*x)*((np.e**(-x**2/2)*x*np.sin(1.5708*x) - 1.5708*np.e**(-x**2/2)*np.cos(1.5708*x) - 5*np.pi*np.sin(5*np.pi*x))/x - (np.cos(5*np.pi*x) - np.e**(-x**2/2)*np.sin(1.5708*x))/x**2) + np.sin(np.pi*x)*((2.4674*np.e**(-x**2/2)*np.sin(1.5708*x) + (np.e**(-x**2/2) - np.e**(-x**2/2)*x**2)*np.sin(1.5708*x) + 3.14159*np.e**(-x**2/2)*x*np.cos(1.5708*x) - 25*np.pi**2*np.cos(5*np.pi*x))/x - (2*(np.e**(-x**2/2)*x * np.sin(1.5708*x) - 1.5708*np.e**(-x**2/2)*np.cos(1.5708*x) - 5*np.pi*np.sin(5*np.pi*x)))/x**2 + (2*(np.cos(5*np.pi*x) - np.e**(-x**2/2)*np.sin(1.5708*x)))/x**3)
# -------------------------------------------------------------------------------


h = 0.1
x = np.linspace(-10, 10, num=200)
y = g(x)
ypr = gpr(x)
ydpr = gdpr(x)

# -------------------------------------------------------------------------------
yprfor = []  # forward euler for g'
yprbac = []  # backward euler for g'
yprcdf = []  # central difference for g'

for i in range(len(x)):
    yprfor.append(forwardeuler(g, x[i], h))
    yprbac.append(backwardeuler(g, x[i], h))
    yprcdf.append(centralDifference(g, x[i], h))
# --------------------------------- 1A ------------------------------------------
plt.title("Function Plot")
plt.plot(x, y, label="given function")
plt.xlabel("x value")
plt.ylabel("y value")
plt.legend()
plt.show()
# ------------------------------------------------------------------------------

# --------------------------------- 1B -----------------------------------------

plt.title("Plots of g'")
plt.plot(x, ypr, label="Analytic")  # Plotting analytic solution
# ------------------------------------------------------------------------------

# --------------------------------- 1C -----------------------------------------
plt.plot(x, yprfor, label="Forward euler")
plt.plot(x, yprbac, label="Backward euler")
plt.plot(x, yprcdf, label="Central difference")
plt.xlabel("x value")
plt.ylabel("Euler Approximations of g'")
plt.legend()
plt.show()
# ------------------------------------------------------------------------------

# --------------------------------- 1D -----------------------------------------

pererrorfor = []  # for percent error in forward euler
pererrorbac = []  # for percent error in backward eulder
pererrorcdf = []  # for percent error in central difference
abserrorfor = []  # for absolute error in forward euler
abserrorbac = []  # for absolute error in backward euler
abserrorcdf = []  # for absolute error in central difference

# computing absolute error at every data point. = |analytic - approx|
for i in range(len(x)):
    abserrorfor.append(np.abs((ypr[i] - yprfor[i])))
    abserrorbac.append(np.abs((ypr[i] - yprbac[i])))
    abserrorcdf.append(np.abs((ypr[i] - yprcdf[i])))

# computing % error at every data point. = 100*(|analytic - approx|/analytic)
for i in range(len(x)):
    pererrorfor.append(100*np.abs((ypr[i] - yprfor[i])/(ypr[i])))
    pererrorbac.append(100*np.abs((ypr[i] - yprbac[i])/(ypr[i])))
    pererrorcdf.append(100*np.abs((ypr[i] - yprcdf[i])/(ypr[i])))

# plotting absolute and percent error
plt.title("Absolute error")
plt.plot(x, abserrorfor, label="Forward euler")
plt.plot(x, abserrorbac, label="Backward euler")
plt.plot(x, abserrorcdf, label="Central difference")
plt.xlabel("x-values")
plt.ylabel("absolute error")
plt.legend()
plt.show()

plt.title("Percentage error")
plt.plot(x, pererrorfor, label="Forward euler")
plt.plot(x, pererrorbac, label="Backward euler")
plt.plot(x, pererrorcdf, label="Central difference")
plt.xlabel("x-values")
plt.ylabel("percentage error")
plt.legend()
plt.show()

# computing the mean error in each of the arrays
forpererror = np.mean(pererrorfor)
bacpererror = np.mean(pererrorbac)
cdfpererror = np.mean(pererrorcdf)
forabserror = np.mean(abserrorfor)
bacabserror = np.mean(abserrorbac)
cdfabserror = np.mean(abserrorcdf)

print("g' Average percent error", forpererror, bacpererror, cdfpererror)
print("g' Average absolute error", forabserror, bacabserror, cdfabserror)
# ------------------------------------------------------------------------------

# ------------------------------- 1E -------------------------------------------
ydprfor = []  # forward euler for g''
ydprbac = []  # backward euler for g''
ydprcdf = []  # central difference for g''

plt.title("Plots of g''")
plt.plot(x, ydpr, label="Analytic")  # plotting the analytic g''

# plotting forward, backward and central difference computations for g''
for i in range(len(x)):
    ydprfor.append(forwardeuler(gpr, x[i], h))
    ydprbac.append(backwardeuler(gpr, x[i], h))
    ydprcdf.append(centralDifference(gpr, x[i], h))


plt.plot(x, ydprfor, label="Forward euler")
plt.plot(x, ydprbac, label="Backward euler")
plt.plot(x, ydprcdf, label="Central difference")
plt.xlabel("x value")
plt.ylabel("Euler Approximations of g''")
plt.legend()
plt.show()

pererrordfor = []  # for percent error in forward euler
pererrordbac = []  # for percent error in backward euler
pererrordcdf = []  # for percent error in centralDifference
abserrordfor = []  # for absolute error in forward euler
abserrordbac = []  # for absolute error in backward euler
abserrordcdf = []  # for absolute error in central difference

# computing absolute error for each data point
for i in range(len(x)):
    abserrordfor.append(np.abs((ydpr[i] - ydprfor[i])))
    abserrordbac.append(np.abs((ydpr[i] - ydprbac[i])))
    abserrordcdf.append(np.abs((ydpr[i] - ydprcdf[i])))

# computing percent error for each data point
for i in range(len(x)):
    pererrordfor.append(100*np.abs((ydpr[i] - ydprfor[i])/(ydpr[i])))
    pererrordbac.append(100*np.abs((ydpr[i] - ydprbac[i])/(ydpr[i])))
    pererrordcdf.append(100*np.abs((ydpr[i] - ydprcdf[i])/(ydpr[i])))

# plotting the errors
plt.title("Absolute error")
plt.plot(x, abserrordfor, label="Forward euler")
plt.plot(x, abserrordbac, label="Backward euler")
plt.plot(x, abserrordcdf, label="Central difference")
plt.xlabel("x-values")
plt.ylabel("absolute error")
plt.legend()
plt.show()

plt.title("Percentage error")
plt.plot(x, pererrordfor, label="Forward euler")
plt.plot(x, pererrordbac, label="Backward euler")
plt.plot(x, pererrordcdf, label="Central difference")
plt.xlabel("x-values")
plt.ylabel("percentage error")
plt.legend()
plt.show()

# computing average percent error and absolute error
dforpererror = np.mean(pererrordfor)
dbacpererror = np.mean(pererrordbac)
dcdfpererror = np.mean(pererrordcdf)
dforabserror = np.mean(abserrordfor)
dbacabserror = np.mean(abserrordbac)
dcdfabserror = np.mean(abserrordcdf)
# ------------------------------------------------------------------------------

# ---------------------------- Print statements --------------------------------
# printing all average errors to console.
print("g'' Average percent error", dforpererror, dbacpererror, dcdfpererror)
print("g'' Average absolute error", dforabserror, dbacabserror, dcdfabserror)
# ------------------------------------------------------------------------------


# ---------------------------- Fourier stuff -----------------------------------
# -------------------------------- 2A -------------------------------------------
N = 4096  # 2^11
x = np.linspace(-10, 10, num=N)  # redefining axis for fourier transformation
f1 = np.array(g(x))  # defining the function in time domain
freqs = np.fft.fftfreq(N)  # defining the frequency axis
mask = freqs > 0  # creating a mask to omit negative frequencies
yf1 = np.array(np.fft.fft(f1))  # performing fft using numpy
reyf1 = 2*(yf1.real/N)  # getting real part of the output after rescaling
imyf1 = 2*(yf1.imag/N)  # getting imaginary part of the output after rescaling
fft_fin1 = 2*np.abs(yf1/N)  # getting the rescaled absolute value


# Plotting

plt.title("space domain")
plt.plot(x, f1, label="actual function")
plt.xlabel("time")
plt.ylabel("g(t)")
plt.legend()
plt.grid()
plt.show()


plt.title("Frequency Domain")
#plt.plot(freqs[mask], reyf1[mask], label="fft_numpy_real")
#plt.plot(freqs[mask], imyf1[mask], label="fft_numpy_imag")
plt.plot(freqs[mask], fft_fin1[mask], label="fft_numpy_abs")
plt.legend()
plt.xlabel("Frequency")
plt.ylabel("g_hat(f)")
plt.grid()
plt.show()
# ------------------------------------------------------------------------------

# ----------------------------------- 2B,C ---------------------------------------
N = 1024  # 2^10
xmin = -10
xmax = 10
step = (xmax-xmin)/(N)
xdata = np.linspace(xmin, xmax, N)  # redefining x in order to do FFT properly
v = g(xdata)
vhat = np.fft.fft(v)  # fft of v
what = 1j*np.zeros(N)  # set up for taking derivative
what[0:int(N/2)] = 1j*np.arange(0, N/2, 1)
what[int(N/2)+1:] = 1j*np.arange(-N/2 + 1, 0, 1)
what = what*vhat
w = np.real(np.fft.ifft(what/4))  # converting back to time domain**
# -------------------------------------------------------------------------------
plt.title("g' using FFT")
plt.plot(xdata, w, label="spectral differentiation")  # FFT output
plt.plot(x, ypr, label="analytic solution")
# Comment the next three lines to compare analytic and spectral solutions
plt.plot(x, yprfor, label="Forward euler")
plt.plot(x, yprbac, label="Backward euler")
plt.plot(x, yprcdf, label="Central difference")
plt.xlabel("time")
plt.ylabel("g'(t)")
plt.legend()
plt.show()  # change num to 4096 in line 39 to make this plot to work.
# ------------------------------------------------------------------------------
# ** We divide by 4 because xdata contains 4096 entries, whereas we are only really calculating 1024.
#   So we need to scale our output down by a factor of 4 in order to normalize

# comment the fft plot above and change num in line 39 to 1024 to make this work.

# we do this because, when we are not plotting, we only want to focus on the
# positive real frequencies, so we need to turn the mask back on to compare
fftforerror = []
fftbacerror = []
fftcdferror = []

print(len(w), len(ypr))

for i in range(len(w)):
    fftforerror.append(np.abs(w[i] - yprfor[i]))
    fftbacerror.append(np.abs(w[i] - yprbac[i]))
    fftcdferror.append(np.abs(w[i] - yprcdf[i]))

print("Spectral vs. Euler errors", np.amax(
    fftforerror), np.amax(fftbacerror), np.amax(fftcdferror))
