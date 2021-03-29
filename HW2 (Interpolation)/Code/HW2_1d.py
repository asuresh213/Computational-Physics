import numpy as np
import random
from scipy import interpolate
import matplotlib.pyplot as plt

#------------------------------------------Data -----------------------------------------------------------------------
x = [-5,-4, -3, -2, -1, 0 ,1, 2, 3,4,5]
ypulse = [0,0,0, 0, 0, 1, 0, 0, 0,0,0]
ystep = [0,0,0, 0, 0, 1, 1, 1, 1,1,1]
xnew = np.linspace(-5,5,endpoint=True)
#----------------------------------------------------------------------------------------------------------------------
fpulse = interpolate.interp1d(x,ypulse, kind = "zero") 
fstep = interpolate.interp1d(x,ystep,kind = "zero")

#-----ask Dr. Pratt about the following!-----

#Note: Regular linear interpolation does not work at x=0 because of the serious discontinuity in the data. 
#Therefore using an zero-th order spline interpolation method preserves the discontinuity at 0.
#In above code, replace "zero" with "linear" for linear interpolation to see the error

#-----------------------------------------------------------------------------------------------------------------------

#----------------------------Plotting-----------------------------------------------------------------------------------
plt.plot(x, ypulse, 'o', label = "pulse data")
plt.plot(x,ystep,'x', label = "step data")
plt.plot(xnew, fpulse(xnew),'--', label = "interpolated pulse")
plt.plot(xnew, fstep(xnew), label = "interpolated step")
plt.legend()
plt.show()
