from numpy import *

def calcdBdu_dp(p,xe):
    xe2=xe[1] 
    xe8=xe[7] 
     
    M = zeros((273,39)) 
     
    M[39,0] = (250*xe8)/261 
    M[40,1] = -(250*xe2)/261 
    M[49,10] = -(250*xe2)/261 
    M[55,16] = -(250*xe2)/261 
    M[61,22] = -(250*xe2)/261 
    M[67,28] = -(250*xe2)/261 
     
    return M 
