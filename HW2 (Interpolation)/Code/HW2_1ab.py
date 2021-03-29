import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

#--------------------------------- Data-------------------------------------------

voltage = [-1.5, -1.0, -0.5, 0.0, 1.0, 2.0, 3.0, 4.0, 4.1, 4.2, 4.5] #given voltage data for x values
current = [-3.375, -1.0, -0.125, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 3.0, 10.0] #given current data for y values
startinterpv = -1.5 #voltage from which we want to interpolate	
cutoffinterpv = 4.5 #voltage until which we want to interpolate
x = [] #Interpolant x values
y = [] #Interpolant y values
index = [] #Dummy array to track the index of the x values in the original voltage array 
temp = 0 #Dummy variable to add to index array after each update 
#----------------------------------------------------------------------------------

for m in voltage:	
	if(m>=startinterpv and m<=cutoffinterpv): #Checking if the x value is in our required range
		x.append(m) 
		index.append(temp) #adding the index value of x value to the index array
	temp+=1 #updating the index

y = current[index[0]:index[-1]+1] #recording the current values at the same indecies as the x's we are interested in

#----------------------------- Creating the interpolation --------------------------

f = interp1d(x,y) #creating a linear 1-d interpolation
f2 = interp1d(x , y, kind="cubic") #creating a cubic 1-d interpolation
xnew = np.linspace(startinterpv, cutoffinterpv,endpoint = "true") #creating the x-values for the interpolated f's

#---------------------------Plotting-------------------------------------------------------

plt.plot(x,y,'o',xnew, f(xnew), '-', xnew, f2(xnew), '--') 
plt.legend(['data','linear','cubic'], loc = 'best')
plt.show()
