from numpy import *

def calcBau(p,xe):
    p1=p[0] 
    p2=p[1] 
    p3=p[2] 
    p4=p[3] 
    p5=p[4] 
    p6=p[5] 
    p7=p[6] 
    p8=p[7] 
    p9=p[8] 
    p10=p[9] 
    p11=p[10] 
    p12=p[11] 
    p13=p[12] 
    p14=p[13] 
    p15=p[14] 
    p16=p[15] 
    p17=p[16] 
    p18=p[17] 
    p19=p[18] 
    p20=p[19] 
    p21=p[20] 
    p22=p[21] 
    p23=p[22] 
    p24=p[23] 
    p25=p[24] 
    p26=p[25] 
    p27=p[26] 
    p28=p[27] 
    p29=p[28] 
    p30=p[29] 
    p31=p[30] 
    p32=p[31] 
    p33=p[32] 
    p34=p[33] 
    p35=p[34] 
    p36=p[35] 
    p37=p[36] 
    p38=p[37] 
    p39=p[38] 
    xe2=xe[1] 
    xe8=xe[7] 
    xe10=xe[9] 
    xe11=xe[10] 
    xe12=xe[11] 
    xe13=xe[12] 
    xe14=xe[13] 
    xe15=xe[14] 
     
    M = zeros((8,39)) 
     
    M[0,0] = -3500*p1*xe8 
    M[2,1] = 3500*p2*xe2 
    M[2,2] = 3500*p3*xe12 
    M[4,2] = -3500*p3*xe12 
    M[2,3] = 3500*p4*xe13 
    M[5,3] = -3500*p4*xe13 
    M[2,4] = 3500*p5*xe14 
    M[6,4] = -3500*p5*xe14 
    M[2,5] = 3500*p6*xe15 
    M[7,5] = -3500*p6*xe15 
    M[3,6] = 3500*p7*xe12 
    M[4,6] = -3500*p7*xe12 
    M[3,7] = 3500*p8*xe13 
    M[5,7] = -3500*p8*xe13 
    M[3,8] = 3500*p9*xe14 
    M[6,8] = -3500*p9*xe14 
    M[3,9] = 3500*p10*xe15 
    M[7,9] = -3500*p10*xe15 
    M[4,10] = 3500*p11*xe2 
    M[2,11] = -3500*p12*xe10 
    M[4,11] = 3500*p12*xe10 
    M[3,12] = -3500*p13*xe11 
    M[4,12] = 3500*p13*xe11 
    M[4,13] = 3500*p14*xe13 
    M[5,13] = -3500*p14*xe13 
    M[4,14] = 3500*p15*xe14 
    M[6,14] = -3500*p15*xe14 
    M[4,15] = 3500*p16*xe15 
    M[7,15] = -3500*p16*xe15 
    M[5,16] = 3500*p17*xe2 
    M[2,17] = -3500*p18*xe10 
    M[5,17] = 3500*p18*xe10 
    M[3,18] = -3500*p19*xe11 
    M[5,18] = 3500*p19*xe11 
    M[4,19] = -3500*p20*xe12 
    M[5,19] = 3500*p20*xe12 
    M[5,20] = 3500*p21*xe14 
    M[6,20] = -3500*p21*xe14 
    M[5,21] = 3500*p22*xe15 
    M[7,21] = -3500*p22*xe15 
    M[6,22] = 3500*p23*xe2 
    M[2,23] = -3500*p24*xe10 
    M[6,23] = 3500*p24*xe10 
    M[3,24] = -3500*p25*xe11 
    M[6,24] = 3500*p25*xe11 
    M[4,25] = -3500*p26*xe12 
    M[6,25] = 3500*p26*xe12 
    M[5,26] = -3500*p27*xe13 
    M[6,26] = 3500*p27*xe13 
    M[6,27] = 3500*p28*xe15 
    M[7,27] = -3500*p28*xe15 
    M[7,28] = 3500*p29*xe2 
    M[2,29] = -3500*p30*xe10 
    M[7,29] = 3500*p30*xe10 
    M[3,30] = -3500*p31*xe11 
    M[7,30] = 3500*p31*xe11 
    M[4,31] = -3500*p32*xe12 
    M[7,31] = 3500*p32*xe12 
    M[5,32] = -3500*p33*xe13 
    M[7,32] = 3500*p33*xe13 
    M[6,33] = -3500*p34*xe14 
    M[7,33] = 3500*p34*xe14 
    M[0,34] = 3500*p35*xe10 
    M[2,34] = -3500*p35*xe10 
    M[0,35] = 3500*p36*xe12 
    M[4,35] = -3500*p36*xe12 
    M[0,36] = 3500*p37*xe13 
    M[5,36] = -3500*p37*xe13 
    M[0,37] = 3500*p38*xe14 
    M[6,37] = -3500*p38*xe14 
    M[0,38] = 3500*p39*xe15 
    M[7,38] = -3500*p39*xe15 
     
    return M 
