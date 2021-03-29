import numpy as np
import random
from scipy import interpolate
import matplotlib.pyplot as plt

#------------------------------------------Data -----------------------------------------------------------------------

x = [-4, -3, -2, -1, 0 ,1, 2, 3, 4, 5]
ypulse = [0, 0, 0, 1, 1, 1, 0, 0, 0, 0]
xnew = np.linspace(-4,5,endpoint=True) #better resolution of the x-values
yextended = [] #array to get more data points for the given pulse function

#----------------------------------------Arrays to calculate errors----------------------------------------------------

errorinterp1 = []
errorinterp2 = []
errorinterp3 = []

#-------------------------------Defining the pulse function to better resolution---------------------------------------

for i in xnew:
	if(i<-1 or i>1):
		yextended.append(0)
	else:
		yextended.append(1)

#----------------------------------------------------------------------------------------------------------------------


f = interpolate.interp1d(x,ypulse, kind = "zero") #zero-th order spline interpolation with JUST THE GIVEN DATA

fprev = interpolate.interp1d(x,ypulse, kind = "nearest") #1-d interpolation (nearest data snap) with JUST THE GIVEN DATA

flag = interpolate.lagrange(x,ypulse) #Lagrange interpolation with JUST THE GIVEN DATA


#--------Calculating error between the interpolations and extended pulse function for better estimation-----------------

for t in range(len(xnew)):
	errorinterp1.append(f(xnew[t]) - yextended[t]) 


for t in range(len(xnew)):
	errorinterp2.append(fprev(xnew[t]) - yextended[t])


for t in range(len(xnew)):
	errorinterp3.append(flag(xnew[t]) - yextended[t])


#-----------------------------------------------------------------------------------------------------------------------



#----------------------------Plotting-----------------------------------------------------------------------------------

#Plotting the interpolated functions

plt.plot(x, ypulse, 'o', label = "Data")
plt.plot(xnew, f(xnew), "--",label = "zero order spline")
plt.plot(xnew, fprev(xnew), label = "1d - nearest snap")
plt.plot(xnew, flag(xnew), label = "lagrange")
plt.legend()
plt.show()

#Plotting the error estimates

plt.plot(xnew, errorinterp1, label = "error zero order spline")
plt.plot(xnew, errorinterp2, label = "error 1d-nearest")
plt.plot(xnew, errorinterp3, label = "error lagrange")
plt.legend()
plt.show()




