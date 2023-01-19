from numpy import *

def J_f_x(x,a,u,d,p):
# auto-generated function from matlab

	u2=u[1]
	u11=u[2]
	u17=u[3]
	u23=u[4]
	u29=u[5]
	p2=p[1]
	p11=p[2]
	p17=p[3]
	p23=p[4]
	p29=p[5]
	
	out1 = - (250*p2*u2)/261 - (250*p11*u11)/261 - (250*p17*u17)/261 - (250*p23*u23)/261 - (250*p29*u29)/261
	
	return out1
