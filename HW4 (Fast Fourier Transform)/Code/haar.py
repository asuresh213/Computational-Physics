import pyfftw
import timeit
import time
import matplotlib.pyplot as plt
import numpy as np
from scipy.fftpack import fft as fourier


# ---------- Defining the Haar function ----------
def Haar(x, t):
    f = []
    for h in x:
        if(h < t or h >= 1+t):
            f.append(0)
        if(h >= t and h < 0.5+t):
            f.append(1)
        if(h >= 0.5+t and h < 1+t):
            f.append(-1)
    return f
# ------------------------------------------------


N = 1024
l = 3
omega = 2.0*np.pi*l
A = 5
x = np.linspace(-l, l, N)

# ---------------Test functions--------------------
#f1 = np.abs(x)
#f2 = np.abs(x)
#f1 = np.e**(-500*x**2)
#f2 = np.e**(-500*(x-0.2)**2)
f1 = 4*np.sin(2*np.pi*A*x) + 3*np.cos(4*np.pi*A*x)
f2 = 4*np.sin(2*np.pi*A*(x-0.25)) + 3*np.cos(4*np.pi*A*(x-0.25))
# -------------------------------------------------
#f1 = np.array(Haar(x, 0))
#f2 = np.array(Haar(x, -2))
freqs = np.fft.fftfreq(N)
mask = freqs > 0
# -------------------- FFT from Numpy -------------
start1 = timeit.timeit()
yf1 = np.array(np.fft.fft(f1))
reyf1 = 2*(yf1.real/N)
imyf1 = 2*(yf1.imag/N)
fft_fin1 = 2*np.abs(yf1/N)
end1 = timeit.timeit()


print("numpy time", np.abs(end1-start1))
# for the actual frequency value of peaks,
# freq = observedpeakvalue*(N/sample length) = observedpeakvalue*(N/2l)

start2 = timeit.timeit()
yf2 = np.fft.fft(f2)
reyf2 = 2*(yf2.real/N)
imyf2 = 2*(yf2.imag/N)
fft_fin2 = 2*np.abs(yf2/N)

end2 = timeit.timeit()
print("numpy time", np.abs(end2-start2))
# -------------------------------------------------


# ----------------- FFT from FFTW -----------------
start3 = timeit.timeit()
fft_object1 = pyfftw.builders.fft(f1)
end3 = timeit.timeit()
b1 = np.array(fft_object1())
reb1 = 2*(b1.real/N)
imb1 = 2*(b1.imag/N)
b_fin1 = 2*np.abs(b1/N)
print("fftw time", np.abs(end3-start3))


start4 = timeit.timeit()
fft_object2 = pyfftw.builders.fft(f2)
end4 = timeit.timeit()
b2 = np.array(fft_object2())
reb2 = 2*(b2.real/N)
imb2 = 2*(b2.imag/N)
b_fin2 = 2*np.abs(b2/N)
print("fftw time", np.abs(end4-start4))
# -------------------------------------------------

plt.figure("space domain")
plt.plot(x, f1, label="actual func")
plt.plot(x, f2, label="shifted function")
plt.legend()
plt.grid()
plt.show()


plt.figure("freq domain numpy")
plt.plot(freqs[mask], reyf1[mask], label="fft_numpy_real")
plt.plot(freqs[mask], imyf1[mask], label="fft_numpy_imag")
plt.plot(freqs[mask], reyf2[mask], label="fft_numpy shifted_real")
plt.plot(freqs[mask], imyf2[mask], label="fft_numpy shifted_imag")
plt.legend()
plt.grid()
plt.show()


plt.figure("freq domain numpy - timeshift")
plt.plot(freqs[mask], fft_fin1[mask], label="fft_numpy shifted_real")
plt.plot(freqs[mask], fft_fin2[mask], label="fft_numpy shifted_imag")
plt.legend()
plt.grid()
plt.show()

plt.figure("freq domain fftw")
plt.plot(freqs[mask], reb1[mask], label="fft_fftw_real")
plt.plot(freqs[mask], imb1[mask], label="fft_fftw_imag")
plt.legend()
plt.grid()
plt.show()


plt.figure("freq domain fftw - timeshift")
plt.plot(freqs[mask], reb2[mask], label="fft_fftw shifted_real")
plt.plot(freqs[mask], imb2[mask], label="fft_fftw shifted_imag")
plt.legend()
plt.grid()
plt.show()
