from numpy import *

def calcBdu(p,xe):
    p1=p[0] 
    p2=p[1] 
    p11=p[10] 
    p17=p[16] 
    p23=p[22] 
    p29=p[28] 
    xe2=xe[1] 
    xe8=xe[7] 
     
    M = zeros((7,39)) 
     
    M[1,0] = (250*p1*xe8)/261 
    M[1,1] = -(250*p2*xe2)/261 
    M[1,10] = -(250*p11*xe2)/261 
    M[1,16] = -(250*p17*xe2)/261 
    M[1,22] = -(250*p23*xe2)/261 
    M[1,28] = -(250*p29*xe2)/261 
     
    return M 
