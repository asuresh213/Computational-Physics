import random
data = []

N = 1000000
#data is comprised of a million random numbers in (0,1). (Mean value = 0.5) 
for i in range(N):
	data.append(random.random())

#--------------------- IMPLIMENTATION OF PSEUDOCODES 1 AND 2 -----------------------------

#classical computation of variance.
def variance(data): 
	sum1 = 0
	sumsq = 0
	for x in data:
		sum1 += x
		sumsq += x**2
	mean = sum1/N
	return ((sumsq - N*mean**2)/(N)) #Note that N*mean = sum1


#Welford 1962
def variance2(data): 
	M = 0
	S = 0
	for k in range(1,N): 
		x = data[k]
		oldM = M
		M += (x-M)/k
		S += (x-M)*(x-oldM)
	return (S/(N))


#==================================== BONUS PROBLEM ==========================================

#Youngs and Cramer 1971

#-----------------------------------PSEUDO-CODE:----------------------------------------------
	
'''

variance(samples):
	m:= 0
	s:=0
	for k from 2 to N:
		x:= samples[k]
		m := m + x
		s := s + (1/k(k-1))(kx-m)^2
	return(s/N)

'''

#---------------------------------Implimentation----------------------------------------------

def variance3(data): 
	T = 0
	P = 0
	for k in range(2,N):
		x = data[k]
		T += x
		P += (1/float(k*(k-1)))*(float(k)*x - T)**2
		
	return (P/N)



v1 = variance(data)
v2 = variance2(data)
v3 = variance3(data)
print(v1, v2, v3)
