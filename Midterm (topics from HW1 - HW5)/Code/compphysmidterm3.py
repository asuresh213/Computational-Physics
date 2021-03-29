import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import interpolate

xfine = np.linspace(1.2, 7.9, num=500)
x = np.array([1.2, 2.8, 4.3, 5.4, 6.8, 7.9])
f = np.array([7.5, 14.1, 38.9, 67.0, 151.8, 270.1])


def curve(a, b, x):
    return a*np.exp(b*x)


# ------------------------------- 1A --------------------------------------------

[res, data] = curve_fit(lambda t, a, b: a*np.exp(b*t), x, f, p0=(4, 0.01))
print(res)
y = curve(res[0], res[1], xfine)
plt.title("Fitting exponentials")
plt.plot(x, f, 'x', label="data")
plt.plot(xfine, y, label="fitted function")
plt.xlabel("x values")
plt.ylabel("f(x) and data")
plt.show()
# -------------------------------------------------------------------------------

# -----------------------------Interpolation -----------------------------------

# directly plotting the cubic spline function using interp1d
fcubic = interpolate.interp1d(x, f, kind="cubic")

# ------------------------------------------------------------------------------
print("f(1.3)", fcubic(1.3), "f(2.9)", fcubic(2.9), "f(4.4)",
      fcubic(4.4), "f(5.5)", fcubic(5.5), "f(6.9)", fcubic(6.9))
# -------------------------------Plotting the Data -----------------------------
plt.plot(x, f, 'o', label="data")
plt.plot(xfine, fcubic(xfine), '--', label="cubic spline")
plt.xlabel("x values")
plt.ylabel("cubic spline and data")
plt.legend()
plt.show()

# ----------------------------------- 3 C ---------------------------------------
datapts = [1.3, 2.9, 4.4, 5.5, 6.9]
abserror = []
pererror = []

for i in range(5):
    abserror.append(np.abs(fcubic(datapts[i]) - curve(res[0], res[1], datapts[i])))
    pererror.append(
        100*np.abs(fcubic(datapts[i]) - curve(res[0], res[1], datapts[i]))/curve(res[0], res[1], datapts[i]))

print("avg abs error:", np.mean(abserror))
print("avg % error", np.mean(pererror))
