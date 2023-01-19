from numpy import *

def J_f_u(x,a,u,d,p):
# auto-generated function from matlab

	x2=x[1]
	a1=a[0]
	p1=p[0]
	p2=p[1]
	p11=p[2]
	p17=p[3]
	p23=p[4]
	p29=p[5]
	
	out1 = (250*a1*p1)/261
	out2 = -(250*p2*x2)/261
	out3 = -(250*p11*x2)/261
	out4 = -(250*p17*x2)/261
	out5 = -(250*p23*x2)/261
	out6 = -(250*p29*x2)/261
	
	return out1, out2, out3, out4, out5, out6
