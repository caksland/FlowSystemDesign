from numpy import *

def calcdFRF_dr(p):
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
     
    M = zeros((1024,39)) 
     
    M[891,0] = p35**2/p1**2 
    M[892,0] = (p35*p36)/p1**2 
    M[893,0] = (p35*p37)/p1**2 
    M[894,0] = (p35*p38)/p1**2 
    M[895,0] = (p35*p39)/p1**2 
    M[923,0] = (p35*p36)/p1**2 
    M[924,0] = p36**2/p1**2 
    M[925,0] = (p36*p37)/p1**2 
    M[926,0] = (p36*p38)/p1**2 
    M[927,0] = (p36*p39)/p1**2 
    M[955,0] = (p35*p37)/p1**2 
    M[956,0] = (p36*p37)/p1**2 
    M[957,0] = p37**2/p1**2 
    M[958,0] = (p37*p38)/p1**2 
    M[959,0] = (p37*p39)/p1**2 
    M[987,0] = (p35*p38)/p1**2 
    M[988,0] = (p36*p38)/p1**2 
    M[989,0] = (p37*p38)/p1**2 
    M[990,0] = p38**2/p1**2 
    M[991,0] = (p38*p39)/p1**2 
    M[1019,0] = (p35*p39)/p1**2 
    M[1020,0] = (p36*p39)/p1**2 
    M[1021,0] = (p37*p39)/p1**2 
    M[1022,0] = (p38*p39)/p1**2 
    M[1023,0] = p39**2/p1**2 
    M[99,1] = p11**2/p2**2 
    M[105,1] = (p11*p17)/p2**2 
    M[111,1] = (p11*p23)/p2**2 
    M[117,1] = (p11*p29)/p2**2 
    M[123,1] = -(p11*p35)/p2**2 
    M[124,1] = -(p11*p36)/p2**2 
    M[125,1] = -(p11*p37)/p2**2 
    M[126,1] = -(p11*p38)/p2**2 
    M[127,1] = -(p11*p39)/p2**2 
    M[291,1] = (p11*p17)/p2**2 
    M[297,1] = p17**2/p2**2 
    M[303,1] = (p17*p23)/p2**2 
    M[309,1] = (p17*p29)/p2**2 
    M[315,1] = -(p17*p35)/p2**2 
    M[316,1] = -(p17*p36)/p2**2 
    M[317,1] = -(p17*p37)/p2**2 
    M[318,1] = -(p17*p38)/p2**2 
    M[319,1] = -(p17*p39)/p2**2 
    M[483,1] = (p11*p23)/p2**2 
    M[489,1] = (p17*p23)/p2**2 
    M[495,1] = p23**2/p2**2 
    M[501,1] = (p23*p29)/p2**2 
    M[507,1] = -(p23*p35)/p2**2 
    M[508,1] = -(p23*p36)/p2**2 
    M[509,1] = -(p23*p37)/p2**2 
    M[510,1] = -(p23*p38)/p2**2 
    M[511,1] = -(p23*p39)/p2**2 
    M[675,1] = (p11*p29)/p2**2 
    M[681,1] = (p17*p29)/p2**2 
    M[687,1] = (p23*p29)/p2**2 
    M[693,1] = p29**2/p2**2 
    M[699,1] = -(p29*p35)/p2**2 
    M[700,1] = -(p29*p36)/p2**2 
    M[701,1] = -(p29*p37)/p2**2 
    M[702,1] = -(p29*p38)/p2**2 
    M[703,1] = -(p29*p39)/p2**2 
    M[867,1] = -(p11*p35)/p2**2 
    M[873,1] = -(p17*p35)/p2**2 
    M[879,1] = -(p23*p35)/p2**2 
    M[885,1] = -(p29*p35)/p2**2 
    M[891,1] = p35**2/p2**2 
    M[892,1] = (p35*p36)/p2**2 
    M[893,1] = (p35*p37)/p2**2 
    M[894,1] = (p35*p38)/p2**2 
    M[895,1] = (p35*p39)/p2**2 
    M[899,1] = -(p11*p36)/p2**2 
    M[905,1] = -(p17*p36)/p2**2 
    M[911,1] = -(p23*p36)/p2**2 
    M[917,1] = -(p29*p36)/p2**2 
    M[923,1] = (p35*p36)/p2**2 
    M[924,1] = p36**2/p2**2 
    M[925,1] = (p36*p37)/p2**2 
    M[926,1] = (p36*p38)/p2**2 
    M[927,1] = (p36*p39)/p2**2 
    M[931,1] = -(p11*p37)/p2**2 
    M[937,1] = -(p17*p37)/p2**2 
    M[943,1] = -(p23*p37)/p2**2 
    M[949,1] = -(p29*p37)/p2**2 
    M[955,1] = (p35*p37)/p2**2 
    M[956,1] = (p36*p37)/p2**2 
    M[957,1] = p37**2/p2**2 
    M[958,1] = (p37*p38)/p2**2 
    M[959,1] = (p37*p39)/p2**2 
    M[963,1] = -(p11*p38)/p2**2 
    M[969,1] = -(p17*p38)/p2**2 
    M[975,1] = -(p23*p38)/p2**2 
    M[981,1] = -(p29*p38)/p2**2 
    M[987,1] = (p35*p38)/p2**2 
    M[988,1] = (p36*p38)/p2**2 
    M[989,1] = (p37*p38)/p2**2 
    M[990,1] = p38**2/p2**2 
    M[991,1] = (p38*p39)/p2**2 
    M[995,1] = -(p11*p39)/p2**2 
    M[1001,1] = -(p17*p39)/p2**2 
    M[1007,1] = -(p23*p39)/p2**2 
    M[1013,1] = -(p29*p39)/p2**2 
    M[1019,1] = (p35*p39)/p2**2 
    M[1020,1] = (p36*p39)/p2**2 
    M[1021,1] = (p37*p39)/p2**2 
    M[1022,1] = (p38*p39)/p2**2 
    M[1023,1] = p39**2/p2**2 
    M[0,2] = p8**2/p3**2 
    M[1,2] = (p8*p9)/p3**2 
    M[2,2] = (p8*p10)/p3**2 
    M[3,2] = (p8*p11)/p3**2 
    M[4,2] = (p8*p12)/p3**2 
    M[6,2] = (p8*p14)/p3**2 
    M[7,2] = (p8*p15)/p3**2 
    M[8,2] = (p8*p16)/p3**2 
    M[11,2] = -(p8*p19)/p3**2 
    M[12,2] = -(p8*p20)/p3**2 
    M[17,2] = -(p8*p25)/p3**2 
    M[18,2] = -(p8*p26)/p3**2 
    M[23,2] = -(p8*p31)/p3**2 
    M[24,2] = -(p8*p32)/p3**2 
    M[28,2] = -(p8*p36)/p3**2 
    M[32,2] = (p8*p9)/p3**2 
    M[33,2] = p9**2/p3**2 
    M[34,2] = (p9*p10)/p3**2 
    M[35,2] = (p9*p11)/p3**2 
    M[36,2] = (p9*p12)/p3**2 
    M[38,2] = (p9*p14)/p3**2 
    M[39,2] = (p9*p15)/p3**2 
    M[40,2] = (p9*p16)/p3**2 
    M[43,2] = -(p9*p19)/p3**2 
    M[44,2] = -(p9*p20)/p3**2 
    M[49,2] = -(p9*p25)/p3**2 
    M[50,2] = -(p9*p26)/p3**2 
    M[55,2] = -(p9*p31)/p3**2 
    M[56,2] = -(p9*p32)/p3**2 
    M[60,2] = -(p9*p36)/p3**2 
    M[64,2] = (p8*p10)/p3**2 
    M[65,2] = (p9*p10)/p3**2 
    M[66,2] = p10**2/p3**2 
    M[67,2] = (p10*p11)/p3**2 
    M[68,2] = (p10*p12)/p3**2 
    M[70,2] = (p10*p14)/p3**2 
    M[71,2] = (p10*p15)/p3**2 
    M[72,2] = (p10*p16)/p3**2 
    M[75,2] = -(p10*p19)/p3**2 
    M[76,2] = -(p10*p20)/p3**2 
    M[81,2] = -(p10*p25)/p3**2 
    M[82,2] = -(p10*p26)/p3**2 
    M[87,2] = -(p10*p31)/p3**2 
    M[88,2] = -(p10*p32)/p3**2 
    M[92,2] = -(p10*p36)/p3**2 
    M[96,2] = (p8*p11)/p3**2 
    M[97,2] = (p9*p11)/p3**2 
    M[98,2] = (p10*p11)/p3**2 
    M[99,2] = p11**2/p3**2 
    M[100,2] = (p11*p12)/p3**2 
    M[102,2] = (p11*p14)/p3**2 
    M[103,2] = (p11*p15)/p3**2 
    M[104,2] = (p11*p16)/p3**2 
    M[107,2] = -(p11*p19)/p3**2 
    M[108,2] = -(p11*p20)/p3**2 
    M[113,2] = -(p11*p25)/p3**2 
    M[114,2] = -(p11*p26)/p3**2 
    M[119,2] = -(p11*p31)/p3**2 
    M[120,2] = -(p11*p32)/p3**2 
    M[124,2] = -(p11*p36)/p3**2 
    M[128,2] = (p8*p12)/p3**2 
    M[129,2] = (p9*p12)/p3**2 
    M[130,2] = (p10*p12)/p3**2 
    M[131,2] = (p11*p12)/p3**2 
    M[132,2] = p12**2/p3**2 
    M[134,2] = (p12*p14)/p3**2 
    M[135,2] = (p12*p15)/p3**2 
    M[136,2] = (p12*p16)/p3**2 
    M[139,2] = -(p12*p19)/p3**2 
    M[140,2] = -(p12*p20)/p3**2 
    M[145,2] = -(p12*p25)/p3**2 
    M[146,2] = -(p12*p26)/p3**2 
    M[151,2] = -(p12*p31)/p3**2 
    M[152,2] = -(p12*p32)/p3**2 
    M[156,2] = -(p12*p36)/p3**2 
    M[192,2] = (p8*p14)/p3**2 
    M[193,2] = (p9*p14)/p3**2 
    M[194,2] = (p10*p14)/p3**2 
    M[195,2] = (p11*p14)/p3**2 
    M[196,2] = (p12*p14)/p3**2 
    M[198,2] = p14**2/p3**2 
    M[199,2] = (p14*p15)/p3**2 
    M[200,2] = (p14*p16)/p3**2 
    M[203,2] = -(p14*p19)/p3**2 
    M[204,2] = -(p14*p20)/p3**2 
    M[209,2] = -(p14*p25)/p3**2 
    M[210,2] = -(p14*p26)/p3**2 
    M[215,2] = -(p14*p31)/p3**2 
    M[216,2] = -(p14*p32)/p3**2 
    M[220,2] = -(p14*p36)/p3**2 
    M[224,2] = (p8*p15)/p3**2 
    M[225,2] = (p9*p15)/p3**2 
    M[226,2] = (p10*p15)/p3**2 
    M[227,2] = (p11*p15)/p3**2 
    M[228,2] = (p12*p15)/p3**2 
    M[230,2] = (p14*p15)/p3**2 
    M[231,2] = p15**2/p3**2 
    M[232,2] = (p15*p16)/p3**2 
    M[235,2] = -(p15*p19)/p3**2 
    M[236,2] = -(p15*p20)/p3**2 
    M[241,2] = -(p15*p25)/p3**2 
    M[242,2] = -(p15*p26)/p3**2 
    M[247,2] = -(p15*p31)/p3**2 
    M[248,2] = -(p15*p32)/p3**2 
    M[252,2] = -(p15*p36)/p3**2 
    M[256,2] = (p8*p16)/p3**2 
    M[257,2] = (p9*p16)/p3**2 
    M[258,2] = (p10*p16)/p3**2 
    M[259,2] = (p11*p16)/p3**2 
    M[260,2] = (p12*p16)/p3**2 
    M[262,2] = (p14*p16)/p3**2 
    M[263,2] = (p15*p16)/p3**2 
    M[264,2] = p16**2/p3**2 
    M[267,2] = -(p16*p19)/p3**2 
    M[268,2] = -(p16*p20)/p3**2 
    M[273,2] = -(p16*p25)/p3**2 
    M[274,2] = -(p16*p26)/p3**2 
    M[279,2] = -(p16*p31)/p3**2 
    M[280,2] = -(p16*p32)/p3**2 
    M[284,2] = -(p16*p36)/p3**2 
    M[352,2] = -(p8*p19)/p3**2 
    M[353,2] = -(p9*p19)/p3**2 
    M[354,2] = -(p10*p19)/p3**2 
    M[355,2] = -(p11*p19)/p3**2 
    M[356,2] = -(p12*p19)/p3**2 
    M[358,2] = -(p14*p19)/p3**2 
    M[359,2] = -(p15*p19)/p3**2 
    M[360,2] = -(p16*p19)/p3**2 
    M[363,2] = p19**2/p3**2 
    M[364,2] = (p19*p20)/p3**2 
    M[369,2] = (p19*p25)/p3**2 
    M[370,2] = (p19*p26)/p3**2 
    M[375,2] = (p19*p31)/p3**2 
    M[376,2] = (p19*p32)/p3**2 
    M[380,2] = (p19*p36)/p3**2 
    M[384,2] = -(p8*p20)/p3**2 
    M[385,2] = -(p9*p20)/p3**2 
    M[386,2] = -(p10*p20)/p3**2 
    M[387,2] = -(p11*p20)/p3**2 
    M[388,2] = -(p12*p20)/p3**2 
    M[390,2] = -(p14*p20)/p3**2 
    M[391,2] = -(p15*p20)/p3**2 
    M[392,2] = -(p16*p20)/p3**2 
    M[395,2] = (p19*p20)/p3**2 
    M[396,2] = p20**2/p3**2 
    M[401,2] = (p20*p25)/p3**2 
    M[402,2] = (p20*p26)/p3**2 
    M[407,2] = (p20*p31)/p3**2 
    M[408,2] = (p20*p32)/p3**2 
    M[412,2] = (p20*p36)/p3**2 
    M[544,2] = -(p8*p25)/p3**2 
    M[545,2] = -(p9*p25)/p3**2 
    M[546,2] = -(p10*p25)/p3**2 
    M[547,2] = -(p11*p25)/p3**2 
    M[548,2] = -(p12*p25)/p3**2 
    M[550,2] = -(p14*p25)/p3**2 
    M[551,2] = -(p15*p25)/p3**2 
    M[552,2] = -(p16*p25)/p3**2 
    M[555,2] = (p19*p25)/p3**2 
    M[556,2] = (p20*p25)/p3**2 
    M[561,2] = p25**2/p3**2 
    M[562,2] = (p25*p26)/p3**2 
    M[567,2] = (p25*p31)/p3**2 
    M[568,2] = (p25*p32)/p3**2 
    M[572,2] = (p25*p36)/p3**2 
    M[576,2] = -(p8*p26)/p3**2 
    M[577,2] = -(p9*p26)/p3**2 
    M[578,2] = -(p10*p26)/p3**2 
    M[579,2] = -(p11*p26)/p3**2 
    M[580,2] = -(p12*p26)/p3**2 
    M[582,2] = -(p14*p26)/p3**2 
    M[583,2] = -(p15*p26)/p3**2 
    M[584,2] = -(p16*p26)/p3**2 
    M[587,2] = (p19*p26)/p3**2 
    M[588,2] = (p20*p26)/p3**2 
    M[593,2] = (p25*p26)/p3**2 
    M[594,2] = p26**2/p3**2 
    M[599,2] = (p26*p31)/p3**2 
    M[600,2] = (p26*p32)/p3**2 
    M[604,2] = (p26*p36)/p3**2 
    M[736,2] = -(p8*p31)/p3**2 
    M[737,2] = -(p9*p31)/p3**2 
    M[738,2] = -(p10*p31)/p3**2 
    M[739,2] = -(p11*p31)/p3**2 
    M[740,2] = -(p12*p31)/p3**2 
    M[742,2] = -(p14*p31)/p3**2 
    M[743,2] = -(p15*p31)/p3**2 
    M[744,2] = -(p16*p31)/p3**2 
    M[747,2] = (p19*p31)/p3**2 
    M[748,2] = (p20*p31)/p3**2 
    M[753,2] = (p25*p31)/p3**2 
    M[754,2] = (p26*p31)/p3**2 
    M[759,2] = p31**2/p3**2 
    M[760,2] = (p31*p32)/p3**2 
    M[764,2] = (p31*p36)/p3**2 
    M[768,2] = -(p8*p32)/p3**2 
    M[769,2] = -(p9*p32)/p3**2 
    M[770,2] = -(p10*p32)/p3**2 
    M[771,2] = -(p11*p32)/p3**2 
    M[772,2] = -(p12*p32)/p3**2 
    M[774,2] = -(p14*p32)/p3**2 
    M[775,2] = -(p15*p32)/p3**2 
    M[776,2] = -(p16*p32)/p3**2 
    M[779,2] = (p19*p32)/p3**2 
    M[780,2] = (p20*p32)/p3**2 
    M[785,2] = (p25*p32)/p3**2 
    M[786,2] = (p26*p32)/p3**2 
    M[791,2] = (p31*p32)/p3**2 
    M[792,2] = p32**2/p3**2 
    M[796,2] = (p32*p36)/p3**2 
    M[896,2] = -(p8*p36)/p3**2 
    M[897,2] = -(p9*p36)/p3**2 
    M[898,2] = -(p10*p36)/p3**2 
    M[899,2] = -(p11*p36)/p3**2 
    M[900,2] = -(p12*p36)/p3**2 
    M[902,2] = -(p14*p36)/p3**2 
    M[903,2] = -(p15*p36)/p3**2 
    M[904,2] = -(p16*p36)/p3**2 
    M[907,2] = (p19*p36)/p3**2 
    M[908,2] = (p20*p36)/p3**2 
    M[913,2] = (p25*p36)/p3**2 
    M[914,2] = (p26*p36)/p3**2 
    M[919,2] = (p31*p36)/p3**2 
    M[920,2] = (p32*p36)/p3**2 
    M[924,2] = p36**2/p3**2 
    M[0,3] = p8**2/p4**2 
    M[6,3] = (p8*p14)/p4**2 
    M[9,3] = -(p8*p17)/p4**2 
    M[10,3] = -(p8*p18)/p4**2 
    M[11,3] = -(p8*p19)/p4**2 
    M[12,3] = -(p8*p20)/p4**2 
    M[13,3] = -(p8*p21)/p4**2 
    M[14,3] = -(p8*p22)/p4**2 
    M[19,3] = (p8*p27)/p4**2 
    M[25,3] = (p8*p33)/p4**2 
    M[29,3] = (p8*p37)/p4**2 
    M[192,3] = (p8*p14)/p4**2 
    M[198,3] = p14**2/p4**2 
    M[201,3] = -(p14*p17)/p4**2 
    M[202,3] = -(p14*p18)/p4**2 
    M[203,3] = -(p14*p19)/p4**2 
    M[204,3] = -(p14*p20)/p4**2 
    M[205,3] = -(p14*p21)/p4**2 
    M[206,3] = -(p14*p22)/p4**2 
    M[211,3] = (p14*p27)/p4**2 
    M[217,3] = (p14*p33)/p4**2 
    M[221,3] = (p14*p37)/p4**2 
    M[288,3] = -(p8*p17)/p4**2 
    M[294,3] = -(p14*p17)/p4**2 
    M[297,3] = p17**2/p4**2 
    M[298,3] = (p17*p18)/p4**2 
    M[299,3] = (p17*p19)/p4**2 
    M[300,3] = (p17*p20)/p4**2 
    M[301,3] = (p17*p21)/p4**2 
    M[302,3] = (p17*p22)/p4**2 
    M[307,3] = -(p17*p27)/p4**2 
    M[313,3] = -(p17*p33)/p4**2 
    M[317,3] = -(p17*p37)/p4**2 
    M[320,3] = -(p8*p18)/p4**2 
    M[326,3] = -(p14*p18)/p4**2 
    M[329,3] = (p17*p18)/p4**2 
    M[330,3] = p18**2/p4**2 
    M[331,3] = (p18*p19)/p4**2 
    M[332,3] = (p18*p20)/p4**2 
    M[333,3] = (p18*p21)/p4**2 
    M[334,3] = (p18*p22)/p4**2 
    M[339,3] = -(p18*p27)/p4**2 
    M[345,3] = -(p18*p33)/p4**2 
    M[349,3] = -(p18*p37)/p4**2 
    M[352,3] = -(p8*p19)/p4**2 
    M[358,3] = -(p14*p19)/p4**2 
    M[361,3] = (p17*p19)/p4**2 
    M[362,3] = (p18*p19)/p4**2 
    M[363,3] = p19**2/p4**2 
    M[364,3] = (p19*p20)/p4**2 
    M[365,3] = (p19*p21)/p4**2 
    M[366,3] = (p19*p22)/p4**2 
    M[371,3] = -(p19*p27)/p4**2 
    M[377,3] = -(p19*p33)/p4**2 
    M[381,3] = -(p19*p37)/p4**2 
    M[384,3] = -(p8*p20)/p4**2 
    M[390,3] = -(p14*p20)/p4**2 
    M[393,3] = (p17*p20)/p4**2 
    M[394,3] = (p18*p20)/p4**2 
    M[395,3] = (p19*p20)/p4**2 
    M[396,3] = p20**2/p4**2 
    M[397,3] = (p20*p21)/p4**2 
    M[398,3] = (p20*p22)/p4**2 
    M[403,3] = -(p20*p27)/p4**2 
    M[409,3] = -(p20*p33)/p4**2 
    M[413,3] = -(p20*p37)/p4**2 
    M[416,3] = -(p8*p21)/p4**2 
    M[422,3] = -(p14*p21)/p4**2 
    M[425,3] = (p17*p21)/p4**2 
    M[426,3] = (p18*p21)/p4**2 
    M[427,3] = (p19*p21)/p4**2 
    M[428,3] = (p20*p21)/p4**2 
    M[429,3] = p21**2/p4**2 
    M[430,3] = (p21*p22)/p4**2 
    M[435,3] = -(p21*p27)/p4**2 
    M[441,3] = -(p21*p33)/p4**2 
    M[445,3] = -(p21*p37)/p4**2 
    M[448,3] = -(p8*p22)/p4**2 
    M[454,3] = -(p14*p22)/p4**2 
    M[457,3] = (p17*p22)/p4**2 
    M[458,3] = (p18*p22)/p4**2 
    M[459,3] = (p19*p22)/p4**2 
    M[460,3] = (p20*p22)/p4**2 
    M[461,3] = (p21*p22)/p4**2 
    M[462,3] = p22**2/p4**2 
    M[467,3] = -(p22*p27)/p4**2 
    M[473,3] = -(p22*p33)/p4**2 
    M[477,3] = -(p22*p37)/p4**2 
    M[608,3] = (p8*p27)/p4**2 
    M[614,3] = (p14*p27)/p4**2 
    M[617,3] = -(p17*p27)/p4**2 
    M[618,3] = -(p18*p27)/p4**2 
    M[619,3] = -(p19*p27)/p4**2 
    M[620,3] = -(p20*p27)/p4**2 
    M[621,3] = -(p21*p27)/p4**2 
    M[622,3] = -(p22*p27)/p4**2 
    M[627,3] = p27**2/p4**2 
    M[633,3] = (p27*p33)/p4**2 
    M[637,3] = (p27*p37)/p4**2 
    M[800,3] = (p8*p33)/p4**2 
    M[806,3] = (p14*p33)/p4**2 
    M[809,3] = -(p17*p33)/p4**2 
    M[810,3] = -(p18*p33)/p4**2 
    M[811,3] = -(p19*p33)/p4**2 
    M[812,3] = -(p20*p33)/p4**2 
    M[813,3] = -(p21*p33)/p4**2 
    M[814,3] = -(p22*p33)/p4**2 
    M[819,3] = (p27*p33)/p4**2 
    M[825,3] = p33**2/p4**2 
    M[829,3] = (p33*p37)/p4**2 
    M[928,3] = (p8*p37)/p4**2 
    M[934,3] = (p14*p37)/p4**2 
    M[937,3] = -(p17*p37)/p4**2 
    M[938,3] = -(p18*p37)/p4**2 
    M[939,3] = -(p19*p37)/p4**2 
    M[940,3] = -(p20*p37)/p4**2 
    M[941,3] = -(p21*p37)/p4**2 
    M[942,3] = -(p22*p37)/p4**2 
    M[947,3] = (p27*p37)/p4**2 
    M[953,3] = (p33*p37)/p4**2 
    M[957,3] = p37**2/p4**2 
    M[33,4] = p9**2/p5**2 
    M[39,4] = (p9*p15)/p5**2 
    M[45,4] = (p9*p21)/p5**2 
    M[47,4] = -(p9*p23)/p5**2 
    M[48,4] = -(p9*p24)/p5**2 
    M[49,4] = -(p9*p25)/p5**2 
    M[50,4] = -(p9*p26)/p5**2 
    M[51,4] = -(p9*p27)/p5**2 
    M[52,4] = -(p9*p28)/p5**2 
    M[58,4] = (p9*p34)/p5**2 
    M[62,4] = (p9*p38)/p5**2 
    M[225,4] = (p9*p15)/p5**2 
    M[231,4] = p15**2/p5**2 
    M[237,4] = (p15*p21)/p5**2 
    M[239,4] = -(p15*p23)/p5**2 
    M[240,4] = -(p15*p24)/p5**2 
    M[241,4] = -(p15*p25)/p5**2 
    M[242,4] = -(p15*p26)/p5**2 
    M[243,4] = -(p15*p27)/p5**2 
    M[244,4] = -(p15*p28)/p5**2 
    M[250,4] = (p15*p34)/p5**2 
    M[254,4] = (p15*p38)/p5**2 
    M[417,4] = (p9*p21)/p5**2 
    M[423,4] = (p15*p21)/p5**2 
    M[429,4] = p21**2/p5**2 
    M[431,4] = -(p21*p23)/p5**2 
    M[432,4] = -(p21*p24)/p5**2 
    M[433,4] = -(p21*p25)/p5**2 
    M[434,4] = -(p21*p26)/p5**2 
    M[435,4] = -(p21*p27)/p5**2 
    M[436,4] = -(p21*p28)/p5**2 
    M[442,4] = (p21*p34)/p5**2 
    M[446,4] = (p21*p38)/p5**2 
    M[481,4] = -(p9*p23)/p5**2 
    M[487,4] = -(p15*p23)/p5**2 
    M[493,4] = -(p21*p23)/p5**2 
    M[495,4] = p23**2/p5**2 
    M[496,4] = (p23*p24)/p5**2 
    M[497,4] = (p23*p25)/p5**2 
    M[498,4] = (p23*p26)/p5**2 
    M[499,4] = (p23*p27)/p5**2 
    M[500,4] = (p23*p28)/p5**2 
    M[506,4] = -(p23*p34)/p5**2 
    M[510,4] = -(p23*p38)/p5**2 
    M[513,4] = -(p9*p24)/p5**2 
    M[519,4] = -(p15*p24)/p5**2 
    M[525,4] = -(p21*p24)/p5**2 
    M[527,4] = (p23*p24)/p5**2 
    M[528,4] = p24**2/p5**2 
    M[529,4] = (p24*p25)/p5**2 
    M[530,4] = (p24*p26)/p5**2 
    M[531,4] = (p24*p27)/p5**2 
    M[532,4] = (p24*p28)/p5**2 
    M[538,4] = -(p24*p34)/p5**2 
    M[542,4] = -(p24*p38)/p5**2 
    M[545,4] = -(p9*p25)/p5**2 
    M[551,4] = -(p15*p25)/p5**2 
    M[557,4] = -(p21*p25)/p5**2 
    M[559,4] = (p23*p25)/p5**2 
    M[560,4] = (p24*p25)/p5**2 
    M[561,4] = p25**2/p5**2 
    M[562,4] = (p25*p26)/p5**2 
    M[563,4] = (p25*p27)/p5**2 
    M[564,4] = (p25*p28)/p5**2 
    M[570,4] = -(p25*p34)/p5**2 
    M[574,4] = -(p25*p38)/p5**2 
    M[577,4] = -(p9*p26)/p5**2 
    M[583,4] = -(p15*p26)/p5**2 
    M[589,4] = -(p21*p26)/p5**2 
    M[591,4] = (p23*p26)/p5**2 
    M[592,4] = (p24*p26)/p5**2 
    M[593,4] = (p25*p26)/p5**2 
    M[594,4] = p26**2/p5**2 
    M[595,4] = (p26*p27)/p5**2 
    M[596,4] = (p26*p28)/p5**2 
    M[602,4] = -(p26*p34)/p5**2 
    M[606,4] = -(p26*p38)/p5**2 
    M[609,4] = -(p9*p27)/p5**2 
    M[615,4] = -(p15*p27)/p5**2 
    M[621,4] = -(p21*p27)/p5**2 
    M[623,4] = (p23*p27)/p5**2 
    M[624,4] = (p24*p27)/p5**2 
    M[625,4] = (p25*p27)/p5**2 
    M[626,4] = (p26*p27)/p5**2 
    M[627,4] = p27**2/p5**2 
    M[628,4] = (p27*p28)/p5**2 
    M[634,4] = -(p27*p34)/p5**2 
    M[638,4] = -(p27*p38)/p5**2 
    M[641,4] = -(p9*p28)/p5**2 
    M[647,4] = -(p15*p28)/p5**2 
    M[653,4] = -(p21*p28)/p5**2 
    M[655,4] = (p23*p28)/p5**2 
    M[656,4] = (p24*p28)/p5**2 
    M[657,4] = (p25*p28)/p5**2 
    M[658,4] = (p26*p28)/p5**2 
    M[659,4] = (p27*p28)/p5**2 
    M[660,4] = p28**2/p5**2 
    M[666,4] = -(p28*p34)/p5**2 
    M[670,4] = -(p28*p38)/p5**2 
    M[833,4] = (p9*p34)/p5**2 
    M[839,4] = (p15*p34)/p5**2 
    M[845,4] = (p21*p34)/p5**2 
    M[847,4] = -(p23*p34)/p5**2 
    M[848,4] = -(p24*p34)/p5**2 
    M[849,4] = -(p25*p34)/p5**2 
    M[850,4] = -(p26*p34)/p5**2 
    M[851,4] = -(p27*p34)/p5**2 
    M[852,4] = -(p28*p34)/p5**2 
    M[858,4] = p34**2/p5**2 
    M[862,4] = (p34*p38)/p5**2 
    M[961,4] = (p9*p38)/p5**2 
    M[967,4] = (p15*p38)/p5**2 
    M[973,4] = (p21*p38)/p5**2 
    M[975,4] = -(p23*p38)/p5**2 
    M[976,4] = -(p24*p38)/p5**2 
    M[977,4] = -(p25*p38)/p5**2 
    M[978,4] = -(p26*p38)/p5**2 
    M[979,4] = -(p27*p38)/p5**2 
    M[980,4] = -(p28*p38)/p5**2 
    M[986,4] = (p34*p38)/p5**2 
    M[990,4] = p38**2/p5**2 
    M[66,5] = p10**2/p6**2 
    M[72,5] = (p10*p16)/p6**2 
    M[78,5] = (p10*p22)/p6**2 
    M[84,5] = (p10*p28)/p6**2 
    M[85,5] = -(p10*p29)/p6**2 
    M[86,5] = -(p10*p30)/p6**2 
    M[87,5] = -(p10*p31)/p6**2 
    M[88,5] = -(p10*p32)/p6**2 
    M[89,5] = -(p10*p33)/p6**2 
    M[90,5] = -(p10*p34)/p6**2 
    M[95,5] = (p10*p39)/p6**2 
    M[258,5] = (p10*p16)/p6**2 
    M[264,5] = p16**2/p6**2 
    M[270,5] = (p16*p22)/p6**2 
    M[276,5] = (p16*p28)/p6**2 
    M[277,5] = -(p16*p29)/p6**2 
    M[278,5] = -(p16*p30)/p6**2 
    M[279,5] = -(p16*p31)/p6**2 
    M[280,5] = -(p16*p32)/p6**2 
    M[281,5] = -(p16*p33)/p6**2 
    M[282,5] = -(p16*p34)/p6**2 
    M[287,5] = (p16*p39)/p6**2 
    M[450,5] = (p10*p22)/p6**2 
    M[456,5] = (p16*p22)/p6**2 
    M[462,5] = p22**2/p6**2 
    M[468,5] = (p22*p28)/p6**2 
    M[469,5] = -(p22*p29)/p6**2 
    M[470,5] = -(p22*p30)/p6**2 
    M[471,5] = -(p22*p31)/p6**2 
    M[472,5] = -(p22*p32)/p6**2 
    M[473,5] = -(p22*p33)/p6**2 
    M[474,5] = -(p22*p34)/p6**2 
    M[479,5] = (p22*p39)/p6**2 
    M[642,5] = (p10*p28)/p6**2 
    M[648,5] = (p16*p28)/p6**2 
    M[654,5] = (p22*p28)/p6**2 
    M[660,5] = p28**2/p6**2 
    M[661,5] = -(p28*p29)/p6**2 
    M[662,5] = -(p28*p30)/p6**2 
    M[663,5] = -(p28*p31)/p6**2 
    M[664,5] = -(p28*p32)/p6**2 
    M[665,5] = -(p28*p33)/p6**2 
    M[666,5] = -(p28*p34)/p6**2 
    M[671,5] = (p28*p39)/p6**2 
    M[674,5] = -(p10*p29)/p6**2 
    M[680,5] = -(p16*p29)/p6**2 
    M[686,5] = -(p22*p29)/p6**2 
    M[692,5] = -(p28*p29)/p6**2 
    M[693,5] = p29**2/p6**2 
    M[694,5] = (p29*p30)/p6**2 
    M[695,5] = (p29*p31)/p6**2 
    M[696,5] = (p29*p32)/p6**2 
    M[697,5] = (p29*p33)/p6**2 
    M[698,5] = (p29*p34)/p6**2 
    M[703,5] = -(p29*p39)/p6**2 
    M[706,5] = -(p10*p30)/p6**2 
    M[712,5] = -(p16*p30)/p6**2 
    M[718,5] = -(p22*p30)/p6**2 
    M[724,5] = -(p28*p30)/p6**2 
    M[725,5] = (p29*p30)/p6**2 
    M[726,5] = p30**2/p6**2 
    M[727,5] = (p30*p31)/p6**2 
    M[728,5] = (p30*p32)/p6**2 
    M[729,5] = (p30*p33)/p6**2 
    M[730,5] = (p30*p34)/p6**2 
    M[735,5] = -(p30*p39)/p6**2 
    M[738,5] = -(p10*p31)/p6**2 
    M[744,5] = -(p16*p31)/p6**2 
    M[750,5] = -(p22*p31)/p6**2 
    M[756,5] = -(p28*p31)/p6**2 
    M[757,5] = (p29*p31)/p6**2 
    M[758,5] = (p30*p31)/p6**2 
    M[759,5] = p31**2/p6**2 
    M[760,5] = (p31*p32)/p6**2 
    M[761,5] = (p31*p33)/p6**2 
    M[762,5] = (p31*p34)/p6**2 
    M[767,5] = -(p31*p39)/p6**2 
    M[770,5] = -(p10*p32)/p6**2 
    M[776,5] = -(p16*p32)/p6**2 
    M[782,5] = -(p22*p32)/p6**2 
    M[788,5] = -(p28*p32)/p6**2 
    M[789,5] = (p29*p32)/p6**2 
    M[790,5] = (p30*p32)/p6**2 
    M[791,5] = (p31*p32)/p6**2 
    M[792,5] = p32**2/p6**2 
    M[793,5] = (p32*p33)/p6**2 
    M[794,5] = (p32*p34)/p6**2 
    M[799,5] = -(p32*p39)/p6**2 
    M[802,5] = -(p10*p33)/p6**2 
    M[808,5] = -(p16*p33)/p6**2 
    M[814,5] = -(p22*p33)/p6**2 
    M[820,5] = -(p28*p33)/p6**2 
    M[821,5] = (p29*p33)/p6**2 
    M[822,5] = (p30*p33)/p6**2 
    M[823,5] = (p31*p33)/p6**2 
    M[824,5] = (p32*p33)/p6**2 
    M[825,5] = p33**2/p6**2 
    M[826,5] = (p33*p34)/p6**2 
    M[831,5] = -(p33*p39)/p6**2 
    M[834,5] = -(p10*p34)/p6**2 
    M[840,5] = -(p16*p34)/p6**2 
    M[846,5] = -(p22*p34)/p6**2 
    M[852,5] = -(p28*p34)/p6**2 
    M[853,5] = (p29*p34)/p6**2 
    M[854,5] = (p30*p34)/p6**2 
    M[855,5] = (p31*p34)/p6**2 
    M[856,5] = (p32*p34)/p6**2 
    M[857,5] = (p33*p34)/p6**2 
    M[858,5] = p34**2/p6**2 
    M[863,5] = -(p34*p39)/p6**2 
    M[994,5] = (p10*p39)/p6**2 
    M[1000,5] = (p16*p39)/p6**2 
    M[1006,5] = (p22*p39)/p6**2 
    M[1012,5] = (p28*p39)/p6**2 
    M[1013,5] = -(p29*p39)/p6**2 
    M[1014,5] = -(p30*p39)/p6**2 
    M[1015,5] = -(p31*p39)/p6**2 
    M[1016,5] = -(p32*p39)/p6**2 
    M[1017,5] = -(p33*p39)/p6**2 
    M[1018,5] = -(p34*p39)/p6**2 
    M[1023,5] = p39**2/p6**2 
    M[0,6] = p8**2/p7**2 
    M[1,6] = (p8*p9)/p7**2 
    M[2,6] = (p8*p10)/p7**2 
    M[5,6] = -(p8*p13)/p7**2 
    M[11,6] = -(p8*p19)/p7**2 
    M[17,6] = -(p8*p25)/p7**2 
    M[23,6] = -(p8*p31)/p7**2 
    M[32,6] = (p8*p9)/p7**2 
    M[33,6] = p9**2/p7**2 
    M[34,6] = (p9*p10)/p7**2 
    M[37,6] = -(p9*p13)/p7**2 
    M[43,6] = -(p9*p19)/p7**2 
    M[49,6] = -(p9*p25)/p7**2 
    M[55,6] = -(p9*p31)/p7**2 
    M[64,6] = (p8*p10)/p7**2 
    M[65,6] = (p9*p10)/p7**2 
    M[66,6] = p10**2/p7**2 
    M[69,6] = -(p10*p13)/p7**2 
    M[75,6] = -(p10*p19)/p7**2 
    M[81,6] = -(p10*p25)/p7**2 
    M[87,6] = -(p10*p31)/p7**2 
    M[160,6] = -(p8*p13)/p7**2 
    M[161,6] = -(p9*p13)/p7**2 
    M[162,6] = -(p10*p13)/p7**2 
    M[165,6] = p13**2/p7**2 
    M[171,6] = (p13*p19)/p7**2 
    M[177,6] = (p13*p25)/p7**2 
    M[183,6] = (p13*p31)/p7**2 
    M[352,6] = -(p8*p19)/p7**2 
    M[353,6] = -(p9*p19)/p7**2 
    M[354,6] = -(p10*p19)/p7**2 
    M[357,6] = (p13*p19)/p7**2 
    M[363,6] = p19**2/p7**2 
    M[369,6] = (p19*p25)/p7**2 
    M[375,6] = (p19*p31)/p7**2 
    M[544,6] = -(p8*p25)/p7**2 
    M[545,6] = -(p9*p25)/p7**2 
    M[546,6] = -(p10*p25)/p7**2 
    M[549,6] = (p13*p25)/p7**2 
    M[555,6] = (p19*p25)/p7**2 
    M[561,6] = p25**2/p7**2 
    M[567,6] = (p25*p31)/p7**2 
    M[736,6] = -(p8*p31)/p7**2 
    M[737,6] = -(p9*p31)/p7**2 
    M[738,6] = -(p10*p31)/p7**2 
    M[741,6] = (p13*p31)/p7**2 
    M[747,6] = (p19*p31)/p7**2 
    M[753,6] = (p25*p31)/p7**2 
    M[759,6] = p31**2/p7**2 
    M[0,7] = 1 
    M[33,8] = 1 
    M[66,9] = 1 
    M[99,10] = 1 
    M[132,11] = 1 
    M[165,12] = 1 
    M[198,13] = 1 
    M[231,14] = 1 
    M[264,15] = 1 
    M[297,16] = 1 
    M[330,17] = 1 
    M[363,18] = 1 
    M[396,19] = 1 
    M[429,20] = 1 
    M[462,21] = 1 
    M[495,22] = 1 
    M[528,23] = 1 
    M[561,24] = 1 
    M[594,25] = 1 
    M[627,26] = 1 
    M[660,27] = 1 
    M[693,28] = 1 
    M[726,29] = 1 
    M[759,30] = 1 
    M[792,31] = 1 
    M[825,32] = 1 
    M[858,33] = 1 
    M[891,34] = 1 
    M[924,35] = 1 
    M[957,36] = 1 
    M[990,37] = 1 
    M[1023,38] = 1 
     
    return M 
