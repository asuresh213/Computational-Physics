import numpy as np
import pywt
import matplotlib.pyplot as plt


def psi(t, a):
    return (2/((np.pi)**(1/4)*np.sqrt(3*a))*(1-(t/a)**2)*np.e**(-(t**2)/(2*a**2)))


def randomfunc(t):
    return np.sin(250 * np.pi * t**2)


waveletname = 'db5'
x = np.linspace(-5, 5, num=2048)  # for psi function
# x = np.linspace(0, 1, num=2048) #for random function
signal = psi(x, 1)
#signal = randomfunc(x)
(cA1, cD1) = pywt.dwt(signal, waveletname, 'smooth')
data = signal
fig, axarr = plt.subplots(nrows=5, ncols=2, figsize=(6, 6))
for ii in range(5):
    (data, coeff_d) = pywt.dwt(data, waveletname)
    axarr[ii, 0].plot(data, 'r')
    axarr[ii, 1].plot(coeff_d, 'g')
    axarr[ii, 0].set_ylabel("Level {}".format(ii + 1), fontsize=14, rotation=90)
    axarr[ii, 0].set_yticklabels([])
    if ii == 0:
        axarr[ii, 0].set_title("Approximation coefficients", fontsize=14)
        axarr[ii, 1].set_title("Detail coefficients", fontsize=14)
    axarr[ii, 1].set_yticklabels([])

reconstructed_signal = pywt.idwt(cA1, cD1, waveletname, 'smooth')

fig, ax = plt.subplots()
plt.plot(x, signal, label='signal')
plt.plot(x, reconstructed_signal, label='reconstructed signal', linestyle='--')
ax.set_xlabel("t")
ax.set_ylabel("psi(t)")
ax.legend(loc='upper left')
plt.show()
