from numpy import *

def calcF(p):
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
     
    M = zeros((39,32)) 
     
    M[2,0] = p8/p3 
    M[3,0] = -p8/p4 
    M[6,0] = -p8/p7 
    M[7,0] = 1 
    M[2,1] = p9/p3 
    M[4,1] = -p9/p5 
    M[6,1] = -p9/p7 
    M[8,1] = 1 
    M[2,2] = p10/p3 
    M[5,2] = -p10/p6 
    M[6,2] = -p10/p7 
    M[9,2] = 1 
    M[1,3] = -p11/p2 
    M[2,3] = p11/p3 
    M[10,3] = 1 
    M[2,4] = p12/p3 
    M[11,4] = 1 
    M[6,5] = p13/p7 
    M[12,5] = 1 
    M[2,6] = p14/p3 
    M[3,6] = -p14/p4 
    M[13,6] = 1 
    M[2,7] = p15/p3 
    M[4,7] = -p15/p5 
    M[14,7] = 1 
    M[2,8] = p16/p3 
    M[5,8] = -p16/p6 
    M[15,8] = 1 
    M[1,9] = -p17/p2 
    M[3,9] = p17/p4 
    M[16,9] = 1 
    M[3,10] = p18/p4 
    M[17,10] = 1 
    M[2,11] = -p19/p3 
    M[3,11] = p19/p4 
    M[6,11] = p19/p7 
    M[18,11] = 1 
    M[2,12] = -p20/p3 
    M[3,12] = p20/p4 
    M[19,12] = 1 
    M[3,13] = p21/p4 
    M[4,13] = -p21/p5 
    M[20,13] = 1 
    M[3,14] = p22/p4 
    M[5,14] = -p22/p6 
    M[21,14] = 1 
    M[1,15] = -p23/p2 
    M[4,15] = p23/p5 
    M[22,15] = 1 
    M[4,16] = p24/p5 
    M[23,16] = 1 
    M[2,17] = -p25/p3 
    M[4,17] = p25/p5 
    M[6,17] = p25/p7 
    M[24,17] = 1 
    M[2,18] = -p26/p3 
    M[4,18] = p26/p5 
    M[25,18] = 1 
    M[3,19] = -p27/p4 
    M[4,19] = p27/p5 
    M[26,19] = 1 
    M[4,20] = p28/p5 
    M[5,20] = -p28/p6 
    M[27,20] = 1 
    M[1,21] = -p29/p2 
    M[5,21] = p29/p6 
    M[28,21] = 1 
    M[5,22] = p30/p6 
    M[29,22] = 1 
    M[2,23] = -p31/p3 
    M[5,23] = p31/p6 
    M[6,23] = p31/p7 
    M[30,23] = 1 
    M[2,24] = -p32/p3 
    M[5,24] = p32/p6 
    M[31,24] = 1 
    M[3,25] = -p33/p4 
    M[5,25] = p33/p6 
    M[32,25] = 1 
    M[4,26] = -p34/p5 
    M[5,26] = p34/p6 
    M[33,26] = 1 
    M[0,27] = p35/p1 
    M[1,27] = p35/p2 
    M[34,27] = 1 
    M[0,28] = p36/p1 
    M[1,28] = p36/p2 
    M[2,28] = -p36/p3 
    M[35,28] = 1 
    M[0,29] = p37/p1 
    M[1,29] = p37/p2 
    M[3,29] = -p37/p4 
    M[36,29] = 1 
    M[0,30] = p38/p1 
    M[1,30] = p38/p2 
    M[4,30] = -p38/p5 
    M[37,30] = 1 
    M[0,31] = p39/p1 
    M[1,31] = p39/p2 
    M[5,31] = -p39/p6 
    M[38,31] = 1 
     
    return M 
