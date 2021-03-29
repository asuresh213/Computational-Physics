import pyfftw
import timeit
import time
import matplotlib.pyplot as plt
import numpy as np
from scipy.fftpack import fft as fourier


# ---------- Defining the Haar function ----------
def sinecardinal(x, t):
    f = []
    for h in x:
        if(h/t == 0):
            f.append(1)
        else:
            f.append(t*np.sin(h/t)/h)
    return f
# ------------------------------------------------


N = 10000
l = 10
x = np.linspace(-l, l, N)

# ---------------Test functions--------------------
#f1 = np.e**(-x**2)
#f2 = np.e**(-500*(x)**2)
#f1 = 4*np.sin(2*np.pi*A*x) + 3*np.cos(4*np.pi*A*x)
#f2 = 4*np.sin(2*np.pi*A*(x-0.25)) + 3*np.cos(4*np.pi*A*(x-0.25))
# -------------------------------------------------
f1 = np.array(sinecardinal(x, 1))
f2 = 0.5*np.array(sinecardinal(x, 2))
freqs = np.fft.fftfreq(N)
mask = freqs > 0
# -------------------- FFT from Numpy -------------

start1 = timeit.timeit()
yf1 = np.array(np.fft.ifft(f1))
reyf1 = 2*(yf1.real/N)
imyf1 = 2*(yf1.imag/N)
fft_fin1 = np.abs(yf1)
end1 = timeit.timeit()


print("numpy time", np.abs(end1-start1))
# for the actual frequency value of peaks,
# freq = observedpeakvalue*(N/sample length) = observedpeakvalue*(N/2l)

start2 = timeit.timeit()
yf2 = np.array(np.fft.ifft(f2))
reyf2 = 2*(yf2.real/N)
imyf2 = 2*(yf2.imag/N)
fft_fin2 = np.abs(yf2)

end2 = timeit.timeit()
print("numpy time", np.abs(end2-start2))
# -------------------------------------------------


# ----------------- FFT from FFTW -----------------
start3 = timeit.timeit()
fft_object1 = pyfftw.builders.ifft(f1)
end3 = timeit.timeit()
b1 = np.array(fft_object1())
reb1 = 2*(b1.real/N)
imb1 = 2*(b1.imag/N)
b_fin1 = np.abs(b1)
print("fftw time", np.abs(end3-start3))


start4 = timeit.timeit()
fft_object2 = pyfftw.builders.ifft(f2)
end4 = timeit.timeit()
b2 = np.array(fft_object2())
reb2 = 2*(b2.real/N)
imb2 = 2*(b2.imag/N)
b_fin2 = np.abs(b2)
print("fftw time", np.abs(end4-start4))
# -------------------------------------------------

plt.figure("space domain")
plt.plot(x, f1, label="actual func")
plt.plot(x, f2, label="scaled function")
plt.legend()
plt.grid()
plt.show()

plt.figure("freq domain absolute value - np.fft")
plt.plot(freqs[mask], fft_fin1[mask], label="abs-value-nonscaled")
plt.plot(freqs[mask], fft_fin2[mask], label="abs-value-scaled")
plt.legend()
plt.grid()
plt.show()

plt.figure("freq domain absolute value - fftw")
plt.plot(freqs[mask], b_fin1[mask], label="abs-value-nonscaled")
plt.plot(freqs[mask], b_fin2[mask], label="abs-value-scaled")
plt.legend()
plt.grid()
plt.show()
