from numpy import *

def calcdBau_dp(p,xe):
    xe2=xe[1] 
    xe8=xe[7] 
    xe10=xe[9] 
    xe11=xe[10] 
    xe12=xe[11] 
    xe13=xe[12] 
    xe14=xe[13] 
    xe15=xe[14] 
     
    M = zeros((312,39)) 
     
    M[0,0] = -3500*xe8 
    M[79,1] = 3500*xe2 
    M[80,2] = 3500*xe12 
    M[158,2] = -3500*xe12 
    M[81,3] = 3500*xe13 
    M[198,3] = -3500*xe13 
    M[82,4] = 3500*xe14 
    M[238,4] = -3500*xe14 
    M[83,5] = 3500*xe15 
    M[278,5] = -3500*xe15 
    M[123,6] = 3500*xe12 
    M[162,6] = -3500*xe12 
    M[124,7] = 3500*xe13 
    M[202,7] = -3500*xe13 
    M[125,8] = 3500*xe14 
    M[242,8] = -3500*xe14 
    M[126,9] = 3500*xe15 
    M[282,9] = -3500*xe15 
    M[166,10] = 3500*xe2 
    M[89,11] = -3500*xe10 
    M[167,11] = 3500*xe10 
    M[129,12] = -3500*xe11 
    M[168,12] = 3500*xe11 
    M[169,13] = 3500*xe13 
    M[208,13] = -3500*xe13 
    M[170,14] = 3500*xe14 
    M[248,14] = -3500*xe14 
    M[171,15] = 3500*xe15 
    M[288,15] = -3500*xe15 
    M[211,16] = 3500*xe2 
    M[95,17] = -3500*xe10 
    M[212,17] = 3500*xe10 
    M[135,18] = -3500*xe11 
    M[213,18] = 3500*xe11 
    M[175,19] = -3500*xe12 
    M[214,19] = 3500*xe12 
    M[215,20] = 3500*xe14 
    M[254,20] = -3500*xe14 
    M[216,21] = 3500*xe15 
    M[294,21] = -3500*xe15 
    M[256,22] = 3500*xe2 
    M[101,23] = -3500*xe10 
    M[257,23] = 3500*xe10 
    M[141,24] = -3500*xe11 
    M[258,24] = 3500*xe11 
    M[181,25] = -3500*xe12 
    M[259,25] = 3500*xe12 
    M[221,26] = -3500*xe13 
    M[260,26] = 3500*xe13 
    M[261,27] = 3500*xe15 
    M[300,27] = -3500*xe15 
    M[301,28] = 3500*xe2 
    M[107,29] = -3500*xe10 
    M[302,29] = 3500*xe10 
    M[147,30] = -3500*xe11 
    M[303,30] = 3500*xe11 
    M[187,31] = -3500*xe12 
    M[304,31] = 3500*xe12 
    M[227,32] = -3500*xe13 
    M[305,32] = 3500*xe13 
    M[267,33] = -3500*xe14 
    M[306,33] = 3500*xe14 
    M[34,34] = 3500*xe10 
    M[112,34] = -3500*xe10 
    M[35,35] = 3500*xe12 
    M[191,35] = -3500*xe12 
    M[36,36] = 3500*xe13 
    M[231,36] = -3500*xe13 
    M[37,37] = 3500*xe14 
    M[271,37] = -3500*xe14 
    M[38,38] = 3500*xe15 
    M[311,38] = -3500*xe15 
     
    return M 