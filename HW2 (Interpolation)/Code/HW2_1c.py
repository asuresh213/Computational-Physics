import numpy as np
import random
from scipy import interpolate
import matplotlib.pyplot as plt

# ------------------------------------------ Data --------------------------------------------------------------------

numdatapoints = 7
x = np.linspace(0, 2*np.pi, num=numdatapoints, endpoint=True)  # create evenly spaced x-values
y = np.sin(x)  # evaluate sin(x) at each of those x values
# create a new set of x-values to plot the interpolated function
xnew = np.linspace(0, 2*np.pi, endpoint=True)
ynew = np.sin(xnew)  # evaluate sin(x) at each of those new x values plot the function

# --------------------------------------------------------------------------------------------------------------------


# ----------------------- optimal soothing value for m data points presented in scipy documentation-------------------
# -----------Note: This smoothing value doesn't work as intendend because of lack of noise and is hence avoided-------
#smoothing = numdatapoints - np.sqrt(2*numdatapoints)


# -----------------------------Interpolation -------------------------------------------------------------------------

# obtaining (t,c,k) values for the unsmoothed biquadratic spline representation
tck = interpolate.splrep(x, y, k=4)
# obtaining (t,c,k) values for the unsmoothed quadratic spline function
tck2 = interpolate.splrep(x, y, k=2)

# plotting the biquadratic spline function from tck
fbiquadratic = interpolate.splev(xnew, tck, der=0)
# directly plotting the cubic spline function using interp1d
fcubic = interpolate.interp1d(x, y, kind="cubic")
# plotting the quadratic spline function from tck2
fquadratic = interpolate.splev(xnew, tck2, der=0)

# --------------------------------------------------------------------------------------------------------------------


# -------------------------------Plotting the Data ------------------------------------------------------------------

plt.plot(x, y, 'o', label="data")
plt.plot(xnew, ynew, label="sin(x)")
plt.plot(xnew, fquadratic, label="quadratic spline")
plt.plot(xnew, fbiquadratic, '.', label="biquadratic spline")
plt.plot(xnew, fcubic(xnew), '--', label="cubic spline")
plt.legend()
plt.show()
