from numpy import *

def f(x,a,u,d,p):
# auto-generated function from matlab

	x1=x[0]
	x2=x[1]
	x3=x[2]
	x4=x[3]
	x5=x[4]
	x6=x[5]
	x7=x[6]
	a1=a[0]
	a2=a[1]
	a3=a[2]
	a4=a[3]
	a5=a[4]
	a6=a[5]
	a7=a[6]
	a8=a[7]
	u1=u[0]
	u2=u[1]
	u11=u[2]
	u17=u[3]
	u23=u[4]
	u29=u[5]
	d3=d[0]
	d4=d[1]
	d5=d[2]
	d6=d[3]
	p1=p[0]
	p2=p[1]
	p11=p[2]
	p17=p[3]
	p23=p[4]
	p29=p[5]
	
	out1 = (16*a1)/15 + (16*a2)/15 - (32*x1)/15
	out2 = (250*a1*p1*u1)/261 - (250*p2*u2*x2)/261 - (250*p11*u11*x2)/261 - (250*p17*u17*x2)/261 - (250*p23*u23*x2)/261 - (250*p29*u29*x2)/261
	out3 = (16*a3)/15 + (16*a4)/15 - (32*x3)/15
	out4 = (16*a5)/3 + d3/375 - (16*x4)/3
	out5 = (16*a6)/3 + d4/375 - (16*x5)/3
	out6 = (16*a7)/3 + d5/375 - (16*x6)/3
	out7 = (16*a8)/3 + d6/375 - (16*x7)/3
	
	return out1, out2, out3, out4, out5, out6, out7
