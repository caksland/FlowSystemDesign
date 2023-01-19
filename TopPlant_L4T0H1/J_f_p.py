from numpy import *

def J_f_p(x,a,u,d,p):
# auto-generated function from matlab

	x2=x[1]
	a1=a[0]
	u1=u[0]
	u2=u[1]
	u11=u[2]
	u17=u[3]
	u23=u[4]
	u29=u[5]
	
	out1 = (250*a1*u1)/261
	out2 = -(250*u2*x2)/261
	out3 = -(250*u11*x2)/261
	out4 = -(250*u17*x2)/261
	out5 = -(250*u23*x2)/261
	out6 = -(250*u29*x2)/261
	
	return out1, out2, out3, out4, out5, out6
