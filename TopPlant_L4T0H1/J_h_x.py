from numpy import *

def J_h_x(x,a,u,d,p):
# auto-generated function from matlab

	u2=u[1]
	u11=u[10]
	u17=u[16]
	u23=u[22]
	u29=u[28]
	p2=p[1]
	p11=p[10]
	p17=p[16]
	p23=p[22]
	p29=p[28]
	
	out1 = 3500*p2*u2
	out2 = 3500*p11*u11
	out3 = 3500*p17*u17
	out4 = 3500*p23*u23
	out5 = 3500*p29*u29
	
	return out1, out2, out3, out4, out5
