from numpy import *

def calcAad(p):
    p11=p[10] 
    p17=p[16] 
    p23=p[22] 
    p29=p[28] 
    p35=p[34] 
    p36=p[35] 
    p37=p[36] 
    p38=p[37] 
    p39=p[38] 
     
    M = zeros((8,7)) 
     
    M[0,0] = 1000 
    M[1,0] = 1000 
    M[2,1] = 175*p35 - 175*p17 - 175*p23 - 175*p29 - 175*p11 + 175*p36 + 175*p37 + 175*p38 + 175*p39 
    M[4,1] = 175*p11 
    M[5,1] = 175*p17 
    M[6,1] = 175*p23 
    M[7,1] = 175*p29 
    M[2,2] = 1000 
    M[3,2] = 1000 
    M[4,3] = 2000 
    M[5,4] = 2000 
    M[6,5] = 2000 
    M[7,6] = 2000 
     
    return M 
