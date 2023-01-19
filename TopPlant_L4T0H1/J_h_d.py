from numpy import *

def J_h_d(x,a,u,d,p):
# auto-generated function from matlab

	a2=a[1]
	d1=d[0]
	d2=d[1]
	
	out1 = 3500*d2 - 3500*a2
	out2 = 3500*d1
	
	return out1, out2
