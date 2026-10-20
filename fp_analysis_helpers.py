# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 14:58:30 2023

WILL EVENTUALLY WANT TO SEPARATE THESE METHODS INTO GENERAL PURPOSE AND RESEARCH PROJECT SPECIFIC

@author: tanne
"""
import init

import pyutils.utils as utils
from sys_neuro_tools import plot_utils, fp_utils
import numpy as np
import pandas as pd
from enum import StrEnum
import matplotlib.pyplot as plt
import scipy.signal as sig
from scipy.optimize import curve_fit
import statsmodels.api as sm
import os.path as path
import os
from glob import glob
from pptx import Presentation
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt
import time
import copy
import warnings
from collections import Counter
import itertools


# %% Session Limiting Lookup Table

# Some sessions have poor signal quality due to patch cord connection issues, so this function is used to exclude fp signal and behavioral data from analyses
# for such sessions, specify time ranges to include in seconds
# these time ranges will be analyzed independently (i.e. separate regressions and baseline calculations)
__sess_independent_ranges = {
    96556: {'PL': [0,3871]},
    101853: {'PL': [0,832]},
    101906: {'PL': [0,1072], 'DMS': [0,2188]},
    102186: {'PL': [[0,349], [350,1901]]},
    102235: {'PL': [[0,57], [57.5,3659], [3659.5]]}, #, 'DMS': [[0,3659], [3659.5]]},
    101729: {'DMS': [[0,1890], [1890.5,3398]]},
    101926: {'PL': [0,542.5]},
    101872: {'PL': [[0,113.5], [114.5,1026.5]]},
    102053: {'PL': [0,2571], 'DMS': [0,2571]},
    102201: {'PL': [0,3442]},
    102340: {'PL': [[0,60.5], [61,1286], [1286.5]]},
    102394: {'PL': [0,3674], 'DMS': [0,3126.5]},
    101717: {'PL': [0,1993]},
    102620: {'PL': [[0,411], [850,1337.5]], 'DMS': [0,849.5]},
    102530: {'PL': [0,925], 'DMS': [0,1692]},
    102580: {'PL': [0,2144], 'DMS': [0,1110.5]},
    102614: {'PL': [[0,2072.5], [2073.5]]},
    102524: {'PL': [0,3718.5]},
    102605: {'PL': [[0,880.5], [881.5]]},
    101896: {'PL': [[0,442.5], [443.5,1220], [1220.5]]},
    102318: {'DMS': [[0,1590.5], [1591]]},
    102367: {'PL': [0,382], 'DMS': [0,382]},
    116589: {'TS': [np.inf]},
    116616: {'PL': [[0,2012.9], [2013.1,2155.7], [2155.9]], 'TS': [0, 5252]}, #
    117649: {'PL': [[0,1931.1], [1931.3]]},
    118992: {'DLS-R': [[0,1166], [1166.1,2862.5], [2862.6,3878], [3878.1]]},
    119000: {'DMS': [0,1649.6], 'PL': [np.inf]},
    119009: {'DLS': [0,3084.9], 'TS': [[0,3182], [3182.1]]}, # [3085.1,3279.6], [3279.7] <- amplitude shrunk too much, by ~1/2
    119194: {'PL': [[0,3828.4], [3828.5]], 'NAc': [[0,3861.6], [3861.7,3914.6], [3914.7]],
             'DMS': [[0,44.4], [44.6,118.8], [119,1929.6], [1929.7,2012.3], [2012.4,2319.9], [2320.1,2895.8], [2896,3036.6], [3036.8,4217.4], [4217.6]]},
    119187: {'DLS-L': [[0,976], [976.1]], 'DLS-R': [[0,2437.1], [2437.2,3645.6]]}, #,[3646.8] <- amplitude shrunk too much, by ~1/2
    119202: {'PL': [[194.5,463.6], [540,1501.3], [1512.2,2050.7], [2081,3511], [3511.2,3754.7], [3768,4114.4], [4120]]},
    119230: {'DLS-L': [[0,1348.9], [1349.1,2821.6], [2821.9,3970.6], [3970.7]]},
    119233: {'DMS': [0,2359]},
    119247: {'PL': [[0,3788.4], [3788.6]], 'NAc': [[0,2485], [2485.1]]},
    119255: {'PL': [[0,3550.3], [3550.5]], 'DLS': [[0,3550.3], [3550.5]]},
    119277: {'DLS-R': [[0,2183.3], [2183.4]]},
    119294: {'DLS': [[0,309.1], [309.2,693.8], [693.9,876.9], [877,1521.3]]}, #, [1521.4]]}, <- amplitude shrunk too much, by ~1/2
    119713: {'NAc': [[0,2001.7], [2001.9,3949.3], [3949.4]]},
    119742: {'TS': [[0,2314.7], [2314.8]]},
    119760: {'TS': [[0,3327.8], [3328]]},
    119785: {'PL': [[0,4248.2], [4248.4,4550.9]], 'DMS': [[0,4164.8], [4164.9]], 'DLS': [0,3965]},
    119793: {'PL': [[0,3641.6], [3641.8]]},
    119830: {'NAc': [[0,2842.3], [2842.5,2863.1], [2863.3,4096.1], [4096.3]], 'DMS': [[0,4120.7], [4120.9]], 'DLS': [[0,4505.4], [4505.6]]},
    119839: {'DLS': [[0,2739.5], [2739.6]]},
    119983: {'TS': [0,3014]},
    120057: {'PL': [0,3922.3], 'NAc': [[0,1712.1], [1712.3,2261.7], [2261.8,3922.3]]},
    120066: {'PL': [[0,3258.4], [3258.6]], 'DMS': [0,3989.5], 'DLS': [[0,1669.8], [1670,2575.8]]},
    120112: {'PL': [[0,3892.2], [3892.4]], 'DMS': [0,3485], 'TS': [[0,3890.4], [3890.5,4285]]},
    124582: {'NAc-L': [[0,130.1], [130.2,219.6], [219.9,345.8], [345.9]], 'NAc-R': [[0,42.5], [42.6,375.7], [375.8,509.6], [511.1,2720], [2720.1]]},
    124598: {'NAc': [[0,90.2], [90.3]], 'TS': [[0,167.3], [167.4]]},
    124605: {'NAc': [[0,849], [849.1]]},
    124606: {'NAc-L': [[0,43], [43.2,69], [69.1,831.6], [832,1022.5], [1022.7,1408.9], [1409,2781.9], [2782,2791.9], [2792]],
             'NAc-R': [[0,131], [131.1,1332.4], [1332.5]]},
    124608: {'PL': [[0,1505.5], [1505.7,1551.5], [1551.6]], 'NAc': [[0,435.1], [435.2,1128], [1128.1,1139.2], [1139.4,1672.4]]},
    124609: {'PL': [[0,391.2], [391.4,1254.7], [1254.8,1864.5], [1864.6,2575.7], [2575.8]]},
    124622: {'NAc-L': [[0,69.1], [69.3,86.3], [87.3,250.8], [250.9,2325.9], [2326,3789.2], [3789.3,3796.6], [3796.8]], 'NAc-R': [[0,337.9], [338]],
             'TS-L': [[0,703.3], [703.4,2581.1], [2581.3]], 'TS-R': [[0,379.9], [380,3790.5], [3790.6,4096.5], [4096.7]]},
    124638: {'TS': [[0,2090.2], [2090.4,2791.1], [2791.3]]},
    124644: {'DMS': [[0,1909.6], [1909.7,2435.7], [2435.9,2864.3], [2864.5,3319.5], [3319.7]], 'DLS': [[0,2281.6], [2281.8]]},
    124655: {'NAc-L': [[0,421], [421.3,998], [998.2,2605.6]], 'NAc-R': [[0,998], [998.2]], 'TS-L': [[0,1888.5], [1888.6]]},
    124664: {'DMS': [[0,33.2], [33.3,48], [48.1,106.2], [106.3,148.9], [149,219.4], [219.5,357.2], [357.3,596.4], [596.6,1106.4], [1106.8,1731],
                     [1732,1756.6], [1756.7,2846.5], [2846.6,3061.1], [3061.3,3067.8], [3067.9]], 'DLS': [[0,2862.7], [2862.8]]},
    124672: {'NAc': [[0,3333.4], [3333.5]]},
    124674: {'NAc': [[0,427.5], [427.7]]},
    124735: {'NAc-L': [[0,3612.7], [3613]], 'NAc-R': [[0,3703], [3703.2]], 'TS-R': [[0,376.6], [376.7]],
             'TS-L': [[0,381.6], [381.7,1487.6], [1487.7,2390.5], [2390.6,3604.1], [3604.3]]},
    124736: {'DMS': [[0,2136.6], [2136.8,2174.1], [2174.2,3176.3], [3176.5,3348.4], [3348.5]], 'TS': [[0,41.7], [41.8,801.9], [802,2136.6], [2136.8,3173.9], [3174]]},
    124737: {'DMS': [[0,1769.4], [1769.5]], 'TS': [[0,517.2], [517.4,1164.5], [1164.6,3160.8], [3160.9,3541], [3541.1]]},
    124746: {'NAc-R': [[0,3093.3], [3093.4]], 'TS-L': [[0,3928.7], [3928.8]]},
    124763: {'NAc': [[0,1240.4], [1240.5]], 'TS': [[0,1602.2], [1602.3,1727.2], [1727.3]]},
    124770: {'DLS': [[9.1,3443], [3443.1]], 'TS': [[0,2506.4], [2506.6]]},
    124771: {'NAc': [[0,164.3], [164.4,1729.6], [1729.7]], 'DMS': [[0,3272.6], [3272.7]],
             'TS': [[0,2649.8], [2650,2818.8], [2819,3855.8], [3855.9]]},
    124772: {'NAc': [[0,3619], [3619.2]]},
    124769: {'NAc-L': [[0,1498.8], [1498.9,1619.8], [1620]], 'NAc-R': [0,4269.6]},
    124924: {'NAc-L': [[0,3463], [3463.1]]},
    124931: {'DMS': [[0,1073], [1073.2,2138.5], [2138.6,2152.3], [2152.4,2808], [2808.1]], 'DLS': [[0,406.9], [407.1,2403.3], [2403.4]],
             'TS': [[0,1073], [1073.2,2152.3], [2152.4,2488.7], [2488.8,2496.6], [2496.8]]},
    124940: {'NAc': [[0,3275.9], [3276.1]], 'DMS': [[0,1998.4], [1998.5,2654.2], [2654.3,3840.9], [3841]]},
    124946: {'DMS': [[0,3922.4], [3922.5]]},
    124948: {'TS-R': [0,2582]},
    124949: {'TS': [[0,2766.5], [2766.6]]},
    124950: {'PL': [[0,125.9], [126]], 'NAc': [[0,262.7], [262.9,1496.7], [1496.9]]},
    124951: {'PL': [[0,3059.9], [3060.1]]},
    124957: {'NAc-L': [[0,3097.6], [3097.7]]},
    124965: {'DLS': [[0,395], [395.1,3350.6], [3350.7]], 'TS': [[0,1355.2], [1355.4,1445], [1445.2,3243.9], [3244,3351.6], [3351.7,3415.9], [3416.1]]},
    124973: {'NAc': [[0,924], [924.1,1727.1], [1727.3]]},
    124992: {'NAc': [[0,4097], [4097.1]], 'DLS': [[0,1992.7], [1992.8,2686], [2686.1,3477.1], [3477.2]], 'TS': [[0,2558.1], [2558.2]]},
    125000: {'TS-L': [[0,180.8], [180.9,396.7], [396.8,414], [414.1,442.1], [442.2,1453], [1453.2]]},
    125006: {'PL': [[0,1949.9], [1950]], 'NAc': [[0,1111.5], [1111.6,2203.4], [2203.6,2946.2], [2946.4]], 'TS': [[0,432], [432.1,3529.5], [3529.6]]},
    125102: {'NAc-L': [[0,3039.3], [3039.4,3170.4], [3170.6,3976.4]], 'TS-R': [[0,3857.4], [3857.6]],
             'TS-L': [[0,1558], [1558.2,3040.7], [3040.9,3957], [3957.2]]},
    125109: {'TS': [[0,1183.8], [1183.9,1661.8], [1661.9,1703], [1703.2]]},
    125118: {'NAc': [[0,2097], [2097.1]], 'DMS': [[0,717.5], [717.7]], 'TS': [[0,38.6], [38.8,543.6], [543.7,718.9], [719,1769.3], [1769.5,3333.6], [3333.7]]},
    125124: {'DLS': [[0,1197.8], [1197.9,1268.2], [1268.4]]},
    125126: {'NAc-L': [[0,363.4], [363.6,370.1], [370.2,1026.3], [1026.4,2695.9], [2696.1,2940.1], [2940.3,2958], [2958.1,3813.5]],
             'TS-L': [[0,1026.3], [1026.4,2695.9], [2696.1,2900.3], [2900.5,2940.1]]},
    125127: {'TS': [[0,1147.1], [1147.3,1408.5], [1408.6,3987.7], [3987.8]]},
    125128: {'NAc': [[0,514.5], [514.7,610.4], [610.5]], 'TS': [[0,805.7], [805.8,3210.2], [3210.3,3248.3], [3248.5,3403.7], [3403.9]]},
    125129: {'NAc': [[0,1614.3], [1614.5]]},
    125136: {'NAc-L': [27.2], 'NAc-R': [[0,357.3], [357.5,2277.4], [2277.6]],
             'TS-L': [[0,223.3], [223.5,1258.9], [1259,1287.9], [1288,2914.8], [2915,2968.7], [2969,3817.4], [3817.6]]}, #
    125146: {'TS': [[0,3368.7], [3368.8]]},
    125164: {'TS-L': [[0,654.5], [654.6]]},
    125171: {'NAc': [[0,2388.3], [2388.5]], 'TS': [[0,466.3], [466.5,2537.3], [2537.4,3487.3], [3487.4]]},
    125179: {'NAc': [[0,466.2], [466.4]], 'TS': [[0,2103.6], [2103.8,2422.7], [2423,3811.1], [3811.3]]},
    129081: {'DLS-L': [[0,338.6], [338.7]]},
    129089: {'TS': [[0,686.4], [686.6,721.8], [722,1464.4], [1464.5,1472.2], [1472.3]]},
    129096: {'NAc': [0,1581.5], 'DMS': [0,373.8], 'DLS': [0,1581.5], 'TS': [[0,862.3], [862.5,1075]]},
    129136: {'DLS': [[0,641.6], [641.7]], 'TS': [0,141.8]},
    129145: {'DMS': [[0,347], [347.1]]},
    129155: {'DMS': [[0,217.1], [217.2,2116], [2116.1,2122.4], [2122.5,2669.1], [2669.2]]},
    129165: {'TS-R': [[0,932.1], [932.2,984.6], [984.7,1052.8], [1053]]},
    129177: {'DLS': [0,1393.5], 'TS': [[0,627.1], [627.2,1393.5], [1393.7]]},
    129179: {'DMS-L': [0,225], 'TS-L': [[0,226.3], [226.4]]},
    129180: {'DLS-R': [[0,1548.7], [1548.8,1964.3], [1964.4]]},
    129195: {'PL': [[0,1320.4], [1320.5,1368.4], [1368.6,1371.8], [1372,1374.1], [1374.2,1382.2], [1382.4,1696.4], [1696.7,2095.2], [2095.4,2217.9], [2218.1]]},
    129201: {'DMS': [[0,580.3], [580.4]]},
    129248: {'TS': [0,3356.4]},
    129265: {'DMS': [[0,402.6], [402.8,2446], [2446.1]]},
    129346: {'NAc': [[0,4064.8], [4064.9]], 'DMS': [[0,2420.8], [2420.9,2787.4], [2787.5,3859.7], [3859.9]]},
    129388: {'DLS-L': [0,3136.6]},
    129405: {'DMS': [0,2132.2], 'TS': [0,2168]},
    129414: {'PL': [[0,1642.7], [1642.8,1746.6], [1746.8]]},
    129432: {'PL': [[0,2064.4], [2064.5,2066.2], [2066.3,2131.3], [2131.4,2146.7], [2146.8,2181.4], [2181.5,2292.8], [2292.9,2302.5], [2304.2]], 'NAc': [0,1746.9],
             'DLS': [0,2507.3], 'TS': [[0,1693.7], [1693.8,1747], [1747.1]]},
    129440: {'NAc-R': [[0,2448.5], [2448.6,2808.4], [2808.5]], 'DLS-R': [0,2453.7]}, # [2453.8,2808.4], [2808.5] <- amplitude shrunk too much, by ~1/2
    129467: {'DLS': [0,2088]},
    129728: {'PL': [[0,84.4], [84.9]], 'TS': [[0,84.4], [84.6]]},
    129737: {'NAc': [[0,1235.9], [1236,1388]], 'DMS': [[0,3061.2], [3061.3]], 'TS': [[0,3061.2], [3061.3]]},
    129747: {'PL': [[0,717.2], [717.4]], 'NAc': [[0,717.2], [717.4]]},
    129772: {'DLS-L': [[0,3564.1], [3564.2]], 'DLS-R': [[0,3043.2], [3043.3]]}, # 'NAc-R': [[0,3043.1], [3043.3]], <- worse baseline artifact if included
    129782: {'PL': [[0,476.6], [476.8,2122], [2123,2182.6], [2183,2321.3], [2321.5]],
             'NAc': [[0,2065.2], [2065.3,2070.5], [2070.6,2906.2], [2906.3,2914.3], [2914.4,3200.8], [3200.9]], 'DLS': [0,2952.8]},
    129791: {'NAc': [[0,144.8], [144.9,248.9], [249]], 'TS': [[0,949.2], [949.3,1326.1], [1326.2]]},
    129818: {'DMS-L': [0,1596.8], 'TS-L': [0,1596.8]},
    129820: {'PL': [[0,2888], [2888.1,2913.2], [2913.3,2992.7], [2992.8]], 'TS': [[0,3004.9], [3005]]},
    129822: {'NAc': [[0,2667.8], [2667.9,3176.7], [3176.8,3220.7], [3220.8,3680.3], [3680.5]], 'DMS': [[0,2664.5], [2664.7,2700], [2700.1,2757.5], [2757.6]],
             'DLS': [[0,3218.3], [3218.4]], 'TS': [0,3317.5]},
    129823: {'PL': [[0,116.1], [116.2,862.3], [862.5,1155.3], [1155.5,2760.9], [2761.1,2780.4], [2780.6]]},
    129824: {'TS-L': [[0,493], [493.1,633.7], [633.8,1163]]},
    129843: {'PL': [[0,3500.4], [3500.5,3513.2], [3513.3]]},
    129862: {'PL': [[0,752.3], [752.4,818.4]], 'NAc': [[0,749.3], [749.4,1022.5], [1022.6,1335.2], [1335.4,1675.6], [1675.7,1860.6], [1860.9]],
             'DMS': [[0,3232.5], [3232.6]], 'TS': [[0,831.6], [831.7,1179.4], [1179.5,1285.6], [1285.7,1572.8], [1573,1599.7], [1599.8,1694], [1694.1,1837], [1837.1,2008.2]]},
    130058: {'PL': [[0,3349.5], [3349.7,3427], [3427.1,3438.9], [3439,3441.3], [3441.4,3497.3], [3497.4,3516.7], [3516.8,3546], [3546.1,3746.6], [3746.7,3783], [3783.1,3822.4], [3822.5]],
             'NAc': [[0,2899.2], [2899.3,2964.4], [2964.5,3187.2], [3187.4,3339.7], [3339.9,3427], [3427.1]],
             'DLS': [[0,3339.6], [3339.7,3395.1], [3395.2,3418.9], [3419,3427], [3427.1]],
             'TS': [[0,2974], [2974.1,3262.5], [3262.7,3339.7], [3339.8,3349.5], [3349.6,3362], [3362.1,3485.1], [3485.2]]},
    130066: {'DMS': [0,2940], 'TS': [[0,2927.4], [2927.5,2965], [2965.2,3519.4], [3519.5]]},
    130075: {'PL': [[0,760], [760.3]]},
    130083: {'DMS-L': [[0,3910], [3910.2]], 'TS-L': [[0,296.8], [296.9]]},
    130128: {'DMS-L': [[0,764.8], [764.9,2252.2], [2252.3,2622.8], [2623,2844.4], [2844.6,2945.1], [2945.3,3039.7], [3039.8,3158.7], [3158.8]],
             'DMS-R': [[0,3103.8], [3103.9,3162.4], [3162.6,3264.3]], 'TS-L': [[0,3006.4], [3006.6,3052.7], [3457]], 'TS-R': [[0,2680.5], [2680.7,2912]]},
    130130: {'PL': [[0,2697.5], [2697.8,3227], [3227.2]], 'NAc': [[0,2660.8], [2660.9]], 'TS': [[0,2691.2], [2691.4]]},
    130131: {'DMS': [[0,1401.5], [1401.7,1908.9]], 'TS': [[0,1625.5], [1625.6,1870.1], [1870.3,1988.2], [1988.3,2264.8], [2265,2677.5]]},
    130154: {'PL': [[0,3528], [3528.2]], 'NAc': [[0,3074.7], [3074.9,3525.9], [3526]], 'TS': [[0,3405.5], [3405.7,3622.9], [3623]]},
    130182: {'TS-R': [[0,3811.6], [3811.7]]},
                            }

# Some sessions have dropout artifacts on some or all channels. These will be set to nan
__sess_dropouts = {
    116543: {'PL': [[1128.2,1128.8], [2303.2,2303.9], [2304.9,2305.4]]},
    116551: {'PL': [[2357.9,2358.1], [3425.9,3427], [3537,3537.8]]},
    116589: {'PL': [[839.7,840.5], [847,847.9], [2189.9,2190.7], [3121.6,3122.2]], 'DMS': [[1037,1037.2], [1297.8,1297.9], [2412.8,2421.9], [2538.3,2539.3], [2547.6,2547.8]]},
    116607: {'all': [3690.7,3691.2]},
    116616: {'PL': [[1007.2,1008.3], [2284,2285.1], [2570.8,2570.9], [2928.5,2929.4], [3445.8,3445.9], [4007.7,4007.9], [4888.6,4889.7]],
             'DMS': [[1645.3,1645.5], [1959.5,1959.7]], 'TS': [[417.7,417.9], [561.2,561.4], [1432.2,1432.4], [3910.8,3911], [4079.4,4079.6]]},
    116668: {'all': [3613,3613.2], 'PL': [[585.6,586.7], [2449.8,2450.9], [3105.3,3105.6], [4418.9,4419.8]], 'TS': [3195.4,3195.6]},
    116828: {'PL': [[50,50.9], [57.4,58.3], [1787,1787.9], [2154.6,2155.1], [3045.2,3045.8], [3249,3250.1], [3438.5,3439.3]], 'TS': [3004.1,3004.3]},
    116914: {'PL': [[124,124.8], [1663.4,1664.4], [2845.7, 2846.7]], 'DMS': [1363.1,1363.3], 'TS': [3224.7,3224.9]},
    117242: {'all': [[494.2,511.8], [564.6,565.2], [568.3,568.5], [569.5,572.2], [761.5,762.2], [765.5,766.4], [771.3,772.2], [792.5,794]], 'DMS': [[1730.8,1731], [3469.5,3469.7]]},
    117251: {'DMS': [3835.5,3835.7], 'TS': [[201.3,201.5], [1168.2,1168.4], [3717,3717.2]]},
    117317: {'DMS': [[836.7,836.9], [1195.3,1195.5], [3403.7,3403.9]], 'TS': [[100.7,100.8], [202.1,202.3], [3201.3, 3201.5]]},
    117442: {'all': [2386.5,2386.6], 'PL': [3576.7,3576.9]},
    117450: {'PL': [[633.7,633.8], [1094.7,1094.9], [1510.3,1511.2], [2508.1,2509.2], [3769,3769.7]], 'DMS': [3839.4,3839.5], 'TS': [[615.2,615.4], [2087.2,2087.3]]},
    117498: {'DMS': [3155.8,3156]},
    117512: {'PL': [[768.2,769.2], [1098.8,1099], [1339.2,1339.4], [1677,1678], [2745,2746], [3461.8,3463]], 'DMS': [1247.5,1247.7],
             'TS': [[625.3,625.5], [1051.4,1051.6], [2692.8,2693], [3493.9,3494.1]]},
    117557: {'DMS': [2473.2,2473.4]},
    117565: {'DMS': [1706.1,1706.3]},
    117615: {'PL': [[2645.7,2645.9], [2869,2870], [2883.2,2884.1]], 'DMS': [3036.6,3036.8], 'TS': [[1910.2,1910.4], [2286.2,2286.4], [3320.8,3321], [3995.5,3995.7]]},
    117649: {'DMS': [44.4,45]},
    117658: {'PL': [1838,1838.2], 'TS': [2701.7,2701.9]},
    118992: {'DLS-L': [2086.4,2086.6], 'all': [778,779.3]},
    119194: {'DMS': [[3081.8,3081.9], [3828.7,3828.9]]},
    119247: {'NAc': [[2499.7,2500.3], [3067.1,3067.5]]},
    119285: {'DMS': [3203.3,3203.9]},
    119785: {'NAc': [[4069.6,4069.9]]},
    119974: {'NAc': [2202.1,2202.3], 'DMS': [3630.4,3630.6]},
    120001: {'DMS': [[1867.2,1867.4], [2176.8,2177]]},
    120066: {'PL': [[3884.5,3885], [3986.1,3986.7], [3989.2,3990], [4002.3,4002.8], [4003.8,4004.5]]},
    120103: {'PL': [[1318.8,1320.6], [1968.6,1969.9], [2196.8,2198.2], [2559.4,2561], [3650.2,3652], [4234.1,4236], [5577.8,5579.4]], 'NAc': [4479.3,4481.4]},
    124598: {'NAc': [1176.5,1176.7]},
    124606: {'NAc-L': [385.6,386.2]},
    124607: {'NAc': [3370.8,3370.9]},
    124655: {'NAc-L': [[1766.8,1767], [1767.6,1767.9]]},
    124735: {'NAc-L': [2798.9,2799.2]},
    124746: {'NAc-L': [3928.7,3929]},
    124769: {'NAc-L': [[3553.8,3554.3], [3871.3,3871.6]]},
    124771: {'NAc': [[2423.9,2424.1], [3481,3482], [3712.9,3713.9]]},
    124940: {'NAc': [[3341,3341.3], [3464.3,3464.7]]},
    124948: {'NAc-R': [[1710.9,1711.5], [1715,1715.5]]},
    124950: {'NAc': [[77.8,78.5]]},
    124957: {'NAc-R': [3097.8,3098.3]},
    124973: {'NAc': [1365.4,1365.7]},
    125000: {'all': [312.3,312.7]},
    125102: {'NAc-L': [[3850.6,3850.9], [3857.3,3857.9]]},
    125109: {'TS': [1879.4,1879.6]},
    125128: {'TS': [2786.9,2787.2]},
    125129: {'PL': [3765.2,3766]},
    125146: {'DLS': [[1853.7,1853.9], [3668.1,3669.7]]},
    125155: {'DMS': [3484,3484.2]},
    125164: {'NAc-L': [344.1,344.3]},
    125179: {'TS': [3813.9,3814.3]},
    129145: {'TS': [256.9,257.6]},
    129248: {'all': [363.3,363.7]},
    129432: {'PL': [2308.1,2308.9], 'NAc': [[481.1,481.4], [484.3,484.6]]},
    129809: {'PL': [3321.8,3322.8]},
    129823: {'PL': [[138,138.3], [399.9,400.6]]},
    130130: {'NAc': [[3387.6,3387.9], [3388.8,3389.1]]},
    130154: {'NAc': [[3622.5,3622.8], [3684.5,3684.9]]},
                   }

__sess_ignore = [100323, 101581, 116498, 116507]

# some regions don't have signal across all sessions, so ignore them entirely
__region_ignore = {198: ['TS']}

# choose 405 over 420 when there are sessions with both for 3.6
preferred_isos = {182: ['405', '420'], 202: ['405', '420'], 179: ['420', '405'],
                  180: ['420', '405'], 188: ['420', '405'], 191: ['420', '405'], 207: ['420', '405']}

default_preferred_isos = ['420', '415', '405']
default_preferred_ligs = ['490', '465']

# %% Loading methods

def load_fp_data(loc_db, subj_sess_ids, isos=None, ligs=None, reload=False, traditional_fit=False, baseline_iso_fit=False, band_iso_fit=False,
                 baseline_band_iso_fit=True, irls_fit=False, baseline_irls_fit=False, tilt_t=False, pre_lpf=10, baseline_lpf=0.0005, iso_bands=[[0,0.5], [0.5,5], [5]],
                 allow_neg_coeffs=False, ignore_sess_start=True, interp_reg_coeffs=False, reprocess=False, save_processed_signals=True,
                 clear_saved_processed_signals=False):
    ''' Loads and processes fiber photometry data, optionally choosing which wavelengths are ligand and isosbestic.'''

    start = time.perf_counter()

    # get fiber photometry data
    fp_data = loc_db.get_sess_fp_data(utils.flatten(subj_sess_ids), reload=reload)

    print('Retreived FP data for {} session(s) in {:.1f} s'.format(len(utils.flatten(subj_sess_ids)), time.perf_counter()-start))

    # separate into different dictionaries
    implant_info = fp_data['implant_info']
    fp_data = fp_data['fp_data']

    # determine which signals we'll be generating
    requested_signal_keys = []
    if traditional_fit:
        requested_signal_keys += ['fitted_iso', 'dff_iso']

    if baseline_iso_fit:
        requested_signal_keys += ['baseline_lig', 'baseline_iso', 'baseline_corr_lig',
                                  'baseline_corr_iso', 'fitted_baseline_iso', 'dff_iso_baseline']
    if band_iso_fit:
        requested_signal_keys += ['fitted_fband_iso', 'dff_iso_fband']

    if baseline_band_iso_fit:
        requested_signal_keys += ['baseline_lig', 'baseline_iso', 'baseline_corr_lig',
                                  'baseline_corr_iso', 'fitted_baseline_fband_iso', 'dff_iso_baseline_fband']
    if irls_fit:
        requested_signal_keys += ['fitted_irls_iso', 'dff_iso_irls']

    if baseline_irls_fit:
        requested_signal_keys += ['fitted_baseline_irls_iso', 'dff_iso_baseline_irls']

    requested_signal_keys = set(requested_signal_keys)

    if ligs is None:
        ligs = default_preferred_ligs
    elif not utils.is_list(ligs):
        ligs = [ligs]

    ligs = np.array(ligs)

    # process the signals
    for subj_id in subj_sess_ids.keys():

        if isos is None:
            subj_isos = preferred_isos.get(subj_id, default_preferred_isos)
        elif not utils.is_list(isos):
            subj_isos = [isos]

        subj_isos = np.array(subj_isos)

        for sess_id in subj_sess_ids[subj_id]:

            reprocessed = False

            # determine if we need to reprocess
            if 'processing_info' not in fp_data[subj_id][sess_id]:
                fp_data[subj_id][sess_id]['processed_signals'] = {}
                force_reprocess = True
            else:
                info = fp_data[subj_id][sess_id]['processing_info']
                force_reprocess = (set(info['ligs']) != set(ligs) or
                                   set(info['isos']) != set(subj_isos) or
                                   info['ignore_sess_start'] != ignore_sess_start or
                                   info['independent_ranges'] != __sess_independent_ranges.get(sess_id, None) or
                                   info['dropouts'] != __sess_dropouts.get(sess_id, None) or
                                   info['pre_lpf'] != pre_lpf or
                                   (info['baseline_lpf'] != baseline_lpf and (baseline_iso_fit or baseline_band_iso_fit)) or
                                   (info['iso_bands'] != iso_bands and (band_iso_fit or baseline_band_iso_fit)) or
                                   (info['tilt_t'] != tilt_t and (traditional_fit or band_iso_fit)) or
                                   info['allow_neg_coeffs'] != allow_neg_coeffs) or reprocess

            raw_signals = fp_data[subj_id][sess_id]['raw_signals']
            sess_processed = fp_data[subj_id][sess_id]['processed_signals']

            subj_regions = list(raw_signals.keys())
            if subj_id in __region_ignore:
                subj_regions = [r for r in subj_regions if r not in __region_ignore[subj_id]]

            for region in subj_regions:

                # determine if we need to reprocess this region
                # get a reference to the processed region signals
                region_processed = sess_processed.setdefault(region, {})

                missing_requested_signal = any(
                    key not in region_processed or region_processed[key] is None
                    for key in requested_signal_keys
                )

                if not (force_reprocess or missing_requested_signal):
                    continue
                else:
                    reprocessed = True

                lig_sel = np.array([k in raw_signals[region].keys() for k in ligs])
                iso_sel = np.array([k in raw_signals[region].keys() for k in subj_isos])

                if sum(lig_sel) > 1:
                    lig = ligs[lig_sel][0]
                    print('Found {} matching ligand wavelengths. Choosing {} nm'.format(sum(lig_sel), ligs[lig_sel][0]))
                elif sum(lig_sel) == 1:
                    lig = ligs[lig_sel][0]
                else:
                    raise Exception('No ligand wavelength found')

                if sum(iso_sel) > 1:
                    iso = subj_isos[iso_sel][0]
                    print('Found {} matching iso wavelengths. Choosing {} nm'.format(sum(iso_sel), subj_isos[iso_sel][0]))
                elif sum(iso_sel) == 1:
                    iso = subj_isos[iso_sel][0]
                else:
                    raise Exception('No isosbestic wavelength found')

                raw_lig = raw_signals[region][lig]
                raw_iso = raw_signals[region][iso]

                start = time.perf_counter()

                new_signals = get_all_processed_signals(raw_lig, raw_iso,
                                                        t=fp_data[subj_id][sess_id]['time'],
                                                        subj_id=subj_id, sess_id=sess_id, region=region,
                                                        trial_start_ts=fp_data[subj_id][sess_id]['trial_start_ts'],
                                                        traditional_fit=traditional_fit,
                                                        baseline_iso_fit=baseline_iso_fit,
                                                        pre_lpf=pre_lpf, baseline_lpf=baseline_lpf,
                                                        band_iso_fit=band_iso_fit, iso_bands=iso_bands,
                                                        baseline_band_iso_fit=baseline_band_iso_fit,
                                                        tilt_t=tilt_t, irls_fit=irls_fit, baseline_irls_fit=baseline_irls_fit,
                                                        allow_neg_coeffs=allow_neg_coeffs,
                                                        ignore_sess_start=ignore_sess_start,
                                                        interp_reg_coeffs=interp_reg_coeffs)

                if force_reprocess:
                    region_processed.clear()

                for key, value in new_signals.items():
                    if value is not None:
                        region_processed[key] = value

                # record which wavelengths were used to process the data
                region_processed['lig'] = lig
                region_processed['iso'] = iso

                print('  Processed FP data in {:.1f} s'.format(time.perf_counter()-start))

            fp_data[subj_id][sess_id]['processing_info'] = {
                'ligs': ligs,
                'isos': subj_isos,
                'ignore_sess_start': ignore_sess_start,
                'independent_ranges': __sess_independent_ranges.get(sess_id, None),
                'dropouts': __sess_dropouts.get(sess_id, None),
                'pre_lpf': pre_lpf,
                'baseline_lpf': baseline_lpf,
                'iso_bands': iso_bands,
                'tilt_t': tilt_t,
                'allow_neg_coeffs': allow_neg_coeffs}

            if save_processed_signals and reprocessed:
                # save processed signals for use later
                loc_db.save_fp_data({subj_id: {sess_id: fp_data[subj_id][sess_id]}}, {subj_id: implant_info[subj_id]})
            elif clear_saved_processed_signals:
                # clear the saved processed signals to free up memory
                fp_data[subj_id][sess_id]['processed_signals'] = {}
                loc_db.save_fp_data({subj_id: {sess_id: fp_data[subj_id][sess_id]}}, {subj_id: implant_info[subj_id]})
                fp_data[subj_id][sess_id]['processed_signals'] = sess_processed

            # recompute z-scores since those are easy to recompute so not necessary to persist
            all_signals = ['dff_iso', 'dff_iso_baseline', 'dff_iso_fband', 'dff_iso_baseline_fband', 'dff_iso_irls', 'dff_iso_baseline_irls']
            for region in subj_regions:
                region_processed = sess_processed[region]

                for sig_name in all_signals:
                    if not region_processed.get(sig_name, None) is None:
                        region_processed['z_'+sig_name] = utils.z_score(region_processed[sig_name])

    return fp_data, implant_info


def get_all_processed_signals(raw_lig, raw_iso, t, subj_id=None, sess_id=None, region=None, trial_start_ts=None, traditional_fit=False,
                              pre_lpf=10, baseline_lpf=0.0005, tilt_t=False, baseline_iso_fit=False, band_iso_fit=False, irls_fit=False,
                              baseline_irls_fit=False, baseline_band_iso_fit=True, iso_bands=[[0,0.5], [0.5,5], [5]], allow_neg_coeffs=False,
                              ignore_sess_start=True, interp_reg_coeffs=False):
    ''' Gets all possible processed signals and intermediaries for the given raw signals.
        Will check to see if any signals should be excluded. Also will also optionally exclude signals before and after the behavior.'''

    # initialize signal variables
    # make a copy of raw signals to handle dropouts
    tmp_raw_lig = raw_lig.astype(np.float32)
    tmp_raw_iso = raw_iso.astype(np.float32)
    empty_signal = np.full_like(tmp_raw_lig, np.nan, dtype=np.float32)
    filtered_lig = empty_signal.copy()
    filtered_iso = empty_signal.copy()

    if baseline_iso_fit or baseline_band_iso_fit:
        baseline_lig = empty_signal.copy()
        baseline_iso = empty_signal.copy()
        baseline_corr_lig = empty_signal.copy()
        baseline_corr_iso = empty_signal.copy()
    else:
        baseline_lig = None
        baseline_iso = None
        baseline_corr_lig = None
        baseline_corr_iso = None

    if traditional_fit:
        fitted_iso = empty_signal.copy()
        dff_iso = empty_signal.copy()
    else:
        fitted_iso = None
        dff_iso = None

    if baseline_iso_fit:
        fitted_baseline_iso = empty_signal.copy()
        dff_iso_baseline = empty_signal.copy()
    else:
        fitted_baseline_iso = None
        dff_iso_baseline = None

    if band_iso_fit:
        fitted_fband_iso = empty_signal.copy()
        dff_iso_fband = empty_signal.copy()
    else:
        fitted_fband_iso = None
        dff_iso_fband = None

    if baseline_band_iso_fit:
        fitted_baseline_fband_iso = empty_signal.copy()
        dff_iso_baseline_fband = empty_signal.copy()
    else:
        fitted_baseline_fband_iso = None
        dff_iso_baseline_fband = None

    if irls_fit:
        fitted_irls_iso = empty_signal.copy()
        dff_iso_irls = empty_signal.copy()
    else:
        fitted_irls_iso = None
        dff_iso_irls = None

    if baseline_irls_fit:
        fitted_baseline_irls_iso = empty_signal.copy()
        dff_iso_baseline_irls = empty_signal.copy()
    else:
        fitted_baseline_irls_iso = None
        dff_iso_baseline_irls = None

    iso_fit_info = {'traditional': [], 'baseline': [], 'fband': [], 'baseline_fband': []}

    # if we are to ignore this session, just leave all processed signals as nan
    ignore_session = (sess_id is not None and sess_id in __sess_ignore) or ((subj_id is not None and region is not None) and region in __region_ignore.get(subj_id, []))

    if not ignore_session:

        sr = 1/np.mean(np.diff(t))

        if not trial_start_ts is None and not t is None and ignore_sess_start:
            # start the signal a little before the first trial start timestamp, with a 1s buffer after the recording starts
            # do this to avoid large photobleaching artifacts
            start_t = max(trial_start_ts[0] - 5, 1)
            start_ts_idx = np.argmin(np.abs(t - start_t))
        else:
            start_ts_idx = 0

        # keep end of recording because there is interesting activity there
        end_ts_idx = len(raw_lig)

        # In order to exclude parts of the signal, need to pass in the session id and region
        if not sess_id is None and not region is None:

            # handle independent time ranges
            if sess_id in __sess_independent_ranges and region in __sess_independent_ranges[sess_id]:

                t_ranges = __sess_independent_ranges[sess_id][region]

                if not utils.is_list(t_ranges[0]):
                    t_ranges = [t_ranges]
            else:
                t_ranges = [[t[0], t[-1]]]

            # handle dropout time ranges
            if sess_id in __sess_dropouts:
                dropout_ranges = []
                if region in __sess_dropouts[sess_id]:
                    ranges = __sess_dropouts[sess_id][region]

                    if not utils.is_list(ranges[0]):
                        ranges = [ranges]

                    dropout_ranges.extend(ranges)

                if 'all' in __sess_dropouts[sess_id]:
                    ranges = __sess_dropouts[sess_id]['all']

                    if not utils.is_list(ranges[0]):
                        ranges = [ranges]

                    dropout_ranges.extend(ranges)

                if len(dropout_ranges) > 0:

                    t_drop = np.full_like(raw_lig, False, dtype=bool)
                    for dropout_range in dropout_ranges:
                        t_sel = (t > dropout_range[0]) & (t < dropout_range[1])
                        t_drop[t_sel] = True

                    tmp_raw_lig[t_drop] = np.nan
                    tmp_raw_iso[t_drop] = np.nan

        else:
            t_ranges = [[t[0], t[-1]]]

        # process each time range independently
        for t_range in t_ranges:

            # if the range starts at infinity, that means ignore the whole signal
            if t_range[0] == np.inf:
                continue

            start_idx = np.argmin(np.abs(t - t_range[0]))
            if len(t_range) == 1:
                # go to the end of the signal
                end_idx = len(t)
            else:
                end_idx = np.argmin(np.abs(t - t_range[1]))+1

            if start_idx < start_ts_idx:
                start_idx = start_ts_idx

            if end_idx > end_ts_idx:
                end_idx = end_ts_idx

            sub_raw_lig = tmp_raw_lig[start_idx:end_idx]
            sub_raw_iso = tmp_raw_iso[start_idx:end_idx]

            sub_filt_lig = fp_utils.filter_signal(sub_raw_lig, pre_lpf, sr)
            sub_filt_iso = fp_utils.filter_signal(sub_raw_iso, pre_lpf, sr)

            filtered_lig[start_idx:end_idx] = sub_filt_lig
            filtered_iso[start_idx:end_idx] = sub_filt_iso

            # traditional isosbestic correction
            if traditional_fit:
                sub_dff_iso, sub_fitted_iso, fit_info = fp_utils.calc_iso_dff(sub_filt_lig, sub_filt_iso, t[start_idx:end_idx], vary_t=tilt_t, allow_neg_coeffs=allow_neg_coeffs)

                iso_fit_info['traditional'].append(fit_info)

                fitted_iso[start_idx:end_idx] = sub_fitted_iso
                dff_iso[start_idx:end_idx] = sub_dff_iso

            # Remove the signal baselines before calculating dF/F
            if baseline_iso_fit or baseline_band_iso_fit or baseline_irls_fit:

                sub_baseline_lig = fp_utils.filter_signal(sub_raw_lig, baseline_lpf, sr, use_multiscale_pad=False)
                sub_baseline_iso = fp_utils.filter_signal(sub_raw_iso, baseline_lpf, sr, use_multiscale_pad=False)

                sub_baseline_corr_lig = sub_filt_lig - sub_baseline_lig
                sub_baseline_corr_iso = sub_filt_iso - sub_baseline_iso

                baseline_lig[start_idx:end_idx] = sub_baseline_lig
                baseline_iso[start_idx:end_idx] = sub_baseline_iso
                baseline_corr_lig[start_idx:end_idx] = sub_baseline_corr_lig
                baseline_corr_iso[start_idx:end_idx] = sub_baseline_corr_iso

                # scale the isosbestic signal to best fit the ligand-dependent signal
                if baseline_iso_fit:
                    sub_fitted_baseline_iso, fit_info = fp_utils.fit_signal(sub_baseline_corr_iso, sub_baseline_corr_lig, t[start_idx:end_idx], vary_t=False, allow_neg_coeffs=allow_neg_coeffs)

                    sub_dff_iso_baseline = ((sub_baseline_corr_lig - sub_fitted_baseline_iso)/(sub_baseline_lig + sub_fitted_baseline_iso)) * 100

                    iso_fit_info['baseline'].append(fit_info)

                    fitted_baseline_iso[start_idx:end_idx] = sub_fitted_baseline_iso
                    dff_iso_baseline[start_idx:end_idx] = sub_dff_iso_baseline

                if baseline_band_iso_fit:
                    sub_fitted_baseline_fband_iso, fit_info = fp_utils.fit_signal(sub_baseline_corr_iso, sub_baseline_corr_lig, t[start_idx:end_idx],
                                                                           vary_t=False, fit_bands=True, f_bands=iso_bands, allow_neg_coeffs=allow_neg_coeffs,
                                                                           interp_coeffs=interp_reg_coeffs)
                    sub_dff_iso_baseline_fband = ((sub_baseline_corr_lig - sub_fitted_baseline_fband_iso)/(sub_baseline_lig + sub_fitted_baseline_fband_iso)) * 100

                    iso_fit_info['baseline_fband'].append(fit_info)

                    fitted_baseline_fband_iso[start_idx:end_idx] = sub_fitted_baseline_fband_iso
                    dff_iso_baseline_fband[start_idx:end_idx] = sub_dff_iso_baseline_fband

                if baseline_irls_fit:
                    not_nans = ~np.isnan(sub_baseline_corr_lig)
                    sub_fitted_baseline_irls_iso = np.full_like(sub_baseline_corr_lig, np.nan)
                    sub_fitted_baseline_irls_iso[not_nans] = sm.RLM(sub_baseline_corr_lig[not_nans], sm.add_constant(sub_baseline_corr_iso[not_nans]),
                                                                    M=sm.robust.norms.TukeyBiweight(c=3)).fit().fittedvalues
                    sub_dff_iso_baseline_irls = ((sub_baseline_corr_lig - sub_fitted_baseline_irls_iso)/(sub_baseline_lig + sub_fitted_baseline_irls_iso)) * 100

                    fitted_baseline_irls_iso[start_idx:end_idx] = sub_fitted_baseline_irls_iso
                    dff_iso_baseline_irls[start_idx:end_idx] = sub_dff_iso_baseline_irls

            if band_iso_fit:
                sub_fitted_fband_iso, fit_info = fp_utils.fit_signal(sub_filt_iso, sub_filt_lig, t[start_idx:end_idx],
                                                              vary_t=tilt_t, fit_bands=True, f_bands=iso_bands, allow_neg_coeffs=allow_neg_coeffs,
                                                              interp_coeffs=interp_reg_coeffs)
                sub_dff_iso_fband = ((sub_filt_lig - sub_fitted_fband_iso)/sub_fitted_fband_iso) * 100

                iso_fit_info['fband'].append(fit_info)

                fitted_fband_iso[start_idx:end_idx] = sub_fitted_fband_iso
                dff_iso_fband[start_idx:end_idx] = sub_dff_iso_fband

            if irls_fit:
                not_nans = ~np.isnan(sub_filt_lig)
                sub_fitted_irls_iso = np.full_like(sub_filt_lig, np.nan)
                sub_fitted_irls_iso[not_nans] = sm.RLM(sub_filt_lig[not_nans], sm.add_constant(sub_filt_iso[not_nans]), M=sm.robust.norms.TukeyBiweight(c=3)).fit().fittedvalues
                sub_dff_iso_irls = ((sub_filt_lig - sub_fitted_irls_iso)/sub_fitted_irls_iso) * 100

                fitted_irls_iso[start_idx:end_idx] = sub_fitted_irls_iso
                dff_iso_irls[start_idx:end_idx] = sub_dff_iso_irls

    return {'raw_lig': tmp_raw_lig,
            'raw_iso': tmp_raw_iso,
            'filtered_lig': filtered_lig,
            'filtered_iso': filtered_iso,
            'baseline_lig': baseline_lig,
            'baseline_iso': baseline_iso,
            'baseline_corr_lig': baseline_corr_lig,
            'baseline_corr_iso': baseline_corr_iso,
            'fitted_iso': fitted_iso,
            'fitted_baseline_iso': fitted_baseline_iso,
            'fitted_fband_iso': fitted_fband_iso,
            'fitted_baseline_fband_iso': fitted_baseline_fband_iso,
            'fitted_irls_iso': fitted_irls_iso,
            'fitted_baseline_irls_iso': fitted_baseline_irls_iso,
            'dff_iso': dff_iso,
            'dff_iso_baseline': dff_iso_baseline,
            'dff_iso_fband': dff_iso_fband,
            'dff_iso_baseline_fband': dff_iso_baseline_fband,
            'dff_iso_irls': dff_iso_irls,
            'dff_iso_baseline_irls': dff_iso_baseline_irls,
            'iso_fit_info': iso_fit_info}

# %% Analysis Methods
exp_decay_form = lambda t, a, tau, b: a*np.exp(-t/tau) + b
exp_decay_bounds = ([     0,      0, -np.inf],
                    [np.inf, np.inf,  np.inf])

def calc_peak_properties(signal, t, filter_params={}, peak_find_params={}, fit_decay=True):
    if len(signal) == 0:
        return {}

    signal = signal.copy()

    dt = np.nanmean(np.diff(t))

    # read params
    filt = filter_params.get('filter', True)
    use_filt_signal = filter_params.get('use_filt_signal_props', False)
    cutoff_f = filter_params.get('cutoff_f', 10)
    peak_tmax = peak_find_params.get('peak_tmax', np.median(t))
    tau_tmax = peak_find_params.get('tau_tmax', t[-1])
    min_dist = peak_find_params.get('min_dist', 0)
    lim_peak_width_to_edges = peak_find_params.get('lim_peak_width_to_edges', True)
    peak_edge_buffer = peak_find_params.get('peak_edge_buffer', 2*dt)
    peak_edge_buffer = int(np.ceil(peak_edge_buffer/dt))

    # smooth signal by filtering
    if filt:
        filt_signal = fp_utils.filter_signal(signal, cutoff_f, 1/dt)
    else:
        filt_signal = signal

    # fill nans so peak finding doesn't have issues
    filt_signal, nan_idxs = fp_utils.fill_signal_nans(filt_signal)

    if use_filt_signal:
        peak_prop_signal = filt_signal
    else:
        peak_prop_signal = signal

    # fill nans again so that masked signals properly handle peak limits
    peak_prop_signal[nan_idxs] = np.nan

    t0_idx = np.argmin(np.abs(t))

    peak_tmax_idx = np.argmin(np.abs(t - peak_tmax))
    tau_tmax_idx = np.argmin(np.abs(t - tau_tmax))

    if ((fit_decay and np.sum(np.isnan(signal[t0_idx:tau_tmax_idx])) > 0.25*(tau_tmax_idx - t0_idx)) or
       (not fit_decay and np.sum(np.isnan(signal[t0_idx:peak_tmax_idx])) > 0.25*(peak_tmax_idx - t0_idx))):
        return {}

    min_peak_dist_idxs = int(min_dist/dt)

    all_peaks, _ = sig.find_peaks(filt_signal)

    peak_idx = None
    if len(all_peaks) > 0:
        peaks = all_peaks[(all_peaks > t0_idx) & (all_peaks < peak_tmax_idx)]
        if len(peaks) > 0:
            # prioritize peaks in the middle of the range
            middle_peaks = peaks[(peaks > t0_idx + peak_edge_buffer) & (peaks < peak_tmax_idx - peak_edge_buffer)]
            if len(middle_peaks) > 0:
                peak_idx = middle_peaks[np.argmax(peak_prop_signal[middle_peaks])]
            else:
                peak_idx = peaks[np.argmax(peak_prop_signal[peaks])]

    if peak_idx is None:
        return {}

    valleys, _ = sig.find_peaks(-filt_signal)

    peak_right_idx = None
    peak_left_idx = None
    if len(valleys) > 0:
        left_valleys = valleys[valleys < peak_idx-min_peak_dist_idxs]
        right_valleys = valleys[(valleys > peak_idx+min_peak_dist_idxs) & (valleys < tau_tmax_idx)]

        if len(left_valleys) > 0:
            peak_left_idx = left_valleys[-1]

            # if there is a sub-peak between the max peak and t=0 that would shorten the calculated peak magnitude,
            # move the peak edge to the next valley until that valley is lower than t=0
            i = -2
            while peak_prop_signal[t0_idx] < peak_prop_signal[peak_left_idx] and len(left_valleys) >= abs(i):
                peak_left_idx = left_valleys[i]
                i -= 1

        if len(right_valleys) > 0:
            peak_right_idx = right_valleys[0]

    if peak_left_idx is None:
        peak_left_idx = t0_idx

    if peak_right_idx is None:
        peak_right_idx = np.nanargmin(peak_prop_signal[peak_idx:tau_tmax_idx]) + peak_idx

    if peak_right_idx >= tau_tmax_idx:
        peak_right_idx = tau_tmax_idx
        lowest_side_idx = peak_left_idx
    else:
        idxs = np.array([peak_left_idx, peak_right_idx])
        lowest_side_idx = idxs[np.nanargmin(peak_prop_signal[idxs])]

    peak_height = peak_prop_signal[peak_idx] - peak_prop_signal[lowest_side_idx]
    peak_time = t[peak_idx]

    # manually find peak width
    peak_width_y = peak_prop_signal[peak_idx] - peak_height*0.5
    y_sel = np.flatnonzero(peak_prop_signal < peak_width_y)
    if lim_peak_width_to_edges:
        left_idxs = y_sel[(y_sel < peak_idx) & (y_sel > peak_left_idx)]
        right_idxs = y_sel[(y_sel > peak_idx) & (y_sel < peak_right_idx)]
    else:
        left_idxs = y_sel[(y_sel < peak_idx)]
        right_idxs = y_sel[(y_sel > peak_idx)]

    if len(left_idxs) > 0:
        left_idx = left_idxs[-1]
    else:
        left_idx = peak_left_idx

    if len(right_idxs) > 0:
        right_idx = right_idxs[0]
    else:
        right_idx = peak_right_idx

    peak_width_left = t[left_idx]
    peak_width_right = t[right_idx]
    peak_width = peak_width_right - peak_width_left

    #     # get peak height and width with peak finding routines
    #     # need to pass in all peak indexes for algorithm to work properly
    #     all_peaks_peak_idx = np.argmin(np.abs(all_peaks - peak_idx))
    #     peak_heights, peak_left_idxs, peak_right_idxs = sig.peak_prominences(avg_signal, all_peaks)
    #     peak_left_idx = peak_left_idxs[all_peaks_peak_idx]
    #     peak_right_idx = peak_right_idxs[all_peaks_peak_idx]
    #     if peak_right_idx > tau_tmax_idx:
    #         peak_right_idx = tau_tmax_idx
    #         lowest_side_idx = peak_left_idx
    #     else:
    #         # overwrite peak height with the height from the lowest peak edge
    #         idxs = np.array([peak_left_idx, peak_right_idx])
    #         lowest_side_idx = idxs[np.argmin(avg_signal[idxs])]

    #     peak_heights[all_peaks_peak_idx] = avg_signal[peak_idx] - avg_signal[lowest_side_idx]
    #     peak_widths, peak_widths_y, peak_widths_left, peak_widths_right = sig.peak_widths(avg_signal, all_peaks, rel_height=0.5,
    #                                                                                      prominence_data=(peak_heights, peak_left_idxs, peak_right_idxs))
    #     peak_height = peak_heights[all_peaks_peak_idx]
    #     peak_width = peak_widths[all_peaks_peak_idx]*dt
    #     peak_width_y = peak_widths_y[all_peaks_peak_idx]
    #     peak_width_left = peak_widths_left[all_peaks_peak_idx]*dt + t[0]
    #     peak_width_right = peak_widths_right[all_peaks_peak_idx]*dt + t[0]


    # if np.random.random() < 0.02: # or peak_height < 0
    #     _, ax = plt.subplots(1,1)
    #     ax.plot(t, signal)
    #     # fill nans before plotting filtered signal
    #     filt_signal[nan_idxs] = np.nan
    #     ax.plot(t, filt_signal)
    #     plot_utils.plot_dashlines([0, peak_tmax], ax=ax)
    #     ax.plot(t[peak_idx], peak_prop_signal[peak_idx], marker=7, markersize=10, color='C1')
    #     ax.vlines(t[peak_idx], peak_prop_signal[peak_idx]-peak_height, peak_prop_signal[peak_idx], color='C2', linestyles='dashed')
    #     ax.plot([t[peak_left_idx], t[peak_right_idx]], [peak_prop_signal[peak_left_idx], peak_prop_signal[peak_right_idx]], color='C2')
    #     ax.hlines(peak_width_y, peak_width_left, peak_width_right, color='C3')

    # fit exponential decay to falling peak
    if fit_decay:
        # add a few extra data points to help the fit be an actual decay
        decay_end_idx = int(peak_idx + 1.2*(peak_right_idx - peak_idx))
        sub_signal = peak_prop_signal[peak_idx:decay_end_idx]
        # fix baseline to be the bottom of the peak
        sub_bounds = copy.copy(exp_decay_bounds)
        decay_baseline = np.min(sub_signal)
        sub_bounds[0][-1] = decay_baseline-0.01 # this needs to be slightly smaller than the upper bound
        sub_bounds[1][-1] = decay_baseline
        decay_params = curve_fit(exp_decay_form, t[peak_idx:decay_end_idx] - peak_time, sub_signal, bounds=sub_bounds)[0]
        decay_form = exp_decay_form
        decay_tau = decay_params[1]
    else:
        decay_tau = np.nan
        decay_form = lambda x: np.nan
        decay_params = []

    return {'peak_time': peak_time, 'peak_height': peak_height, 'peak_width': peak_width,
            'peak_width_info': {'y': peak_width_y, 't_lims': [peak_width_left, peak_width_right]},
            'peak_end_time': t[peak_right_idx], 'decay_tau': decay_tau, 'decay_params': decay_params, 'decay_form': decay_form}


def calc_iqr_multiple(table, group_by_cols, parameters):
    table = table.copy()
    if not utils.is_list(group_by_cols):
        group_by_cols = [group_by_cols]

    if not utils.is_list(parameters):
        parameters = [parameters]

    iqr_mult_keys = {param: 'iqr_mult_' + param for param in parameters}
    # initialize iqr columns
    for param in parameters:
        if not iqr_mult_keys[param] in table:
            table[iqr_mult_keys[param]] = np.nan

    groupings = table[group_by_cols].drop_duplicates().to_numpy()
    for group in groupings:
        sel = np.all(np.array([(table[group_col] == group_val) for group_col, group_val in zip(group_by_cols, group)]), axis=0)

        sub_table = table.loc[sel]
        for param in parameters:
            q1, q3 = np.nanquantile(sub_table[param], [0.25, 0.75])
            iqr = q3 - q1
            below_sel = sub_table[param] < q1
            above_sel = sub_table[param] > q3
            sub_table.loc[below_sel, iqr_mult_keys[param]] = (sub_table.loc[below_sel, param] - q1)/iqr
            sub_table.loc[above_sel, iqr_mult_keys[param]] = (sub_table.loc[above_sel, param] - q3)/iqr

        table.loc[sel, iqr_mult_keys.values()] = sub_table[iqr_mult_keys.values()]

    return table



# %% Plotting Methods

def view_processed_signals(processed_signals, t, dec=10, title='Full Signals', vert_marks=[], plot_traditional=True, plot_baseline=True,
                           plot_fband=True, plot_baseline_fband=True, plot_irls=False, plot_baseline_irls=False, t_min=0, t_max=np.inf, region_order=None):

    regions = list(processed_signals.keys())

    if not region_order is None:
        regions = sort_subj_regions(regions, region_order)

    n_rows = len(regions)

    t_sel = (t >= t_min) & (t <= t_max)
    dec_idxs = np.flatnonzero(t_sel)[::dec]
    plot_sel = np.full(t.shape, False, dtype=bool)
    plot_sel[dec_idxs] = True

    plot_t = t[plot_sel]

    if len(vert_marks) > 0:
        vert_marks = vert_marks[(vert_marks > t_min) & (vert_marks < t_max)]

    # declare panel information
    panels = []
    panels.append({
            'title': 'Raw Signals',
            'ylabel': 'Fluorescent Signal (V)',
            'legend_loc': 'center right',
            'signals': [
                {'key': 'raw_lig', 'label': 'Raw Lig', 'color': 'C0', 'alpha': 0.5},
                {'key': 'baseline_lig', 'label': 'Lig Baseline', 'color': 'C0', 'linestyle': '--', 'enabled': plot_baseline or plot_baseline_fband},
                {'key': 'raw_iso', 'label': 'Raw Iso', 'color': 'C1', 'alpha': 0.5},
                {'key': 'baseline_iso', 'label': 'Iso Baseline', 'color': 'C1', 'linestyle': '--', 'enabled': plot_baseline or plot_baseline_fband}
            ]
        })

    if plot_traditional or plot_fband or plot_irls:
        panels.append({
                'title': 'Iso Fit Signals',
                'ylabel': 'Fluorescent Signal (V)',
                'legend_loc': 'upper right',
                'signals': [
                    {'key': 'filtered_lig', 'label': 'Filtered Lig', 'alpha': 0.5},
                    {'key': 'fitted_iso', 'label': 'Fitted Iso', 'alpha': 0.5, 'enabled': plot_traditional},
                    {'key': 'fitted_fband_iso', 'label': 'Fitted F-band Iso', 'alpha': 0.5, 'enabled': plot_fband},
                    {'key': 'fitted_irls_iso', 'label': 'Fitted IRLS Iso', 'alpha': 0.5, 'enabled': plot_irls}
                ]
            })

    if plot_baseline or plot_baseline_fband or plot_baseline_irls:
        panels.append({
                'title': 'Baseline Subtracted Signals',
                'ylabel': 'Fluorescent Signal (dV)',
                'legend_loc': 'upper right',
                'signals': [
                    {'key': 'baseline_corr_lig', 'label': 'Baseline Corrected Lig', 'alpha': 0.5},
                    {'key': 'baseline_corr_iso', 'label': 'Baseline Corrected Iso', 'alpha': 0.5},
                    {'key': 'fitted_baseline_iso', 'label': 'Fitted Baseline Corrected Iso', 'alpha': 0.5, 'enabled': plot_baseline},
                    {'key': 'fitted_baseline_fband_iso', 'label': 'Fitted Baseline F-band Iso', 'alpha': 0.5, 'enabled': plot_baseline_fband},
                    {'key': 'fitted_baseline_irls_iso', 'label': 'Fitted Baseline IRLS Iso', 'alpha': 0.5, 'enabled': plot_baseline_irls}
                ]
            })

    panels.append({
            'title': 'dF/F Signals',
            'ylabel': '% dF/F',
            'legend_loc': 'upper right',
            'signals': [
                {'key': 'dff_iso', 'label': 'Iso dF/F', 'alpha': 0.5, 'enabled': plot_traditional},
                {'key': 'dff_iso_baseline', 'label': 'Baseline Corrected dF/F', 'alpha': 0.5, 'enabled': plot_baseline},
                {'key': 'dff_iso_fband', 'label': 'F-band dF/F', 'alpha': 0.5, 'enabled': plot_fband},
                {'key': 'dff_iso_baseline_fband', 'label': 'Baseline F-band dF/F', 'alpha': 0.5, 'enabled': plot_baseline_fband},
                {'key': 'dff_iso_irls', 'label': 'IRLS dF/F', 'alpha': 0.5, 'enabled': plot_irls},
                {'key': 'dff_iso_baseline_irls', 'label': 'Baseline IRLS dF/F', 'alpha': 0.5, 'enabled': plot_baseline_irls},

            ]
        })


    # do plotting
    fig, axs = plt.subplots(n_rows, len(panels), layout='constrained', figsize=[9*len(panels), 4*n_rows], sharex=True)

    if n_rows == 1:
        axs = axs[None, :]

    plt.suptitle(title)

    for i, region in enumerate(regions):
        sub_signals = processed_signals[region]

        gen_title = region + ' {}'

        for j, panel in enumerate(panels):
            ax = axs[i, j]

            plot_utils.plot_dashlines(vert_marks, ax=ax)

            for signal_info in panel['signals']:
                if not signal_info.get('enabled', True):
                    continue

                sig_key = signal_info['key']
                if sig_key not in sub_signals or sub_signals[sig_key] is None:
                    continue

                plot_y = sub_signals[sig_key][plot_sel]

                plot_kwargs = {
                    k: v for k, v in signal_info.items()
                    if k not in ['key', 'label', 'enabled']
                }

                ax.plot(plot_t, plot_y, label=signal_info['label'], **plot_kwargs)[0]

            ax.set_title(gen_title.format(panel['title']))
            ax.set_xlabel('Time (s)')
            ax.set_ylabel(panel['ylabel'])
            ax.xaxis.set_tick_params(which='both', labelbottom=True)

            ax.legend(loc=panel['legend_loc'])

    return fig


def view_signal(processed_signals, t, signal_type, title=None, dec=10, vert_marks=[],
                filter_outliers=False, outlier_zthresh=10, t_min=0, t_max=np.inf, figsize=None,
                ylabel=''):

    if utils.is_dict(list(processed_signals.values())[0]):
        n_rows = len(processed_signals.values())
    else:
        n_rows = 1
        # make a temporary outer dictionary for ease of use with for loop
        processed_signals = {'temp': processed_signals}

    t = t[::dec]

    t_min_idx = np.argmax(t > t_min)
    t_max_idx = np.argwhere(t < t_max)[-1,0]

    if title is None:
        title = signal_type

    if figsize is None:
        figsize = (7, 3*n_rows)

    _, axs = plt.subplots(n_rows, 1, layout='constrained', figsize=figsize)
    plt.suptitle(title)

    if len(vert_marks) > 0:
        vert_marks = vert_marks[(vert_marks > t_min) & (vert_marks < t_max)]

    for i, (sub_key, sub_signals) in enumerate(processed_signals.items()):
        # remove outliers in z-score space
        filt_sel = np.full(t.shape, True)
        if filter_outliers:
            # filter based on raw signals
            z_sig = utils.z_score(sub_signals[signal_type][::dec])
            filt_sel = filt_sel & (np.abs(z_sig) < outlier_zthresh)

        # repurpose outlier filter for time filter as well
        filt_sel[:t_min_idx] = False
        filt_sel[t_max_idx:] = False

        filt_t = t[filt_sel]

        # plot raw signals and baseline
        ax = axs[i]
        ax.plot(filt_t, sub_signals[signal_type][::dec][filt_sel])
        plot_utils.plot_dashlines(vert_marks, ax=ax)
        ax.set_title(sub_key)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel(ylabel)


# def plot_aligned_signals(signal_dict, t, title, sub_titles_dict, x_label, y_label,
#                          cmap = 'viridis', outlier_thresh=10, trial_markers=None):
#     ''' Plot aligned signals in the given nested dictionary where the first set of keys define the rows and the
#         second set of keys in the dictionaries indexed by the first set of keys define the columns

#         In each 'cell' will plot a heatmap of the aligned signals above an average signal trace
#     '''

#     outer_keys = [k for k in signal_dict.keys() if k != 't']
#     n_rows = len(outer_keys)
#     n_cols = len(signal_dict[outer_keys[0]])

#     fig, axs = plt.subplots(n_rows*2, n_cols, height_ratios=np.tile([3,2], n_rows),
#                             figsize=(5*n_cols, 4*n_rows), layout='constrained')

#     plt.suptitle(title)

#     for i, key in enumerate(outer_keys):

#         # remove outliers in z-score space
#         z_signals = [utils.z_score(signal) for signal in signal_dict[key].values() if len(signal) > 0]
#         signals_no_outliers = {}
#         for j, (name, signal) in enumerate(signal_dict[key].items()):
#             if len(signal) > 0:
#                 signal_no_outliers = signal.copy()
#                 signal_no_outliers[np.abs(z_signals[j]) > outlier_thresh] = np.nan
#                 signals_no_outliers[name] = signal_no_outliers
#             else:
#                 signals_no_outliers[name] = signal

#         max_act = np.max([np.max(signal[~np.isnan(signal)]) for signal in signals_no_outliers.values() if len(signal) > 0])
#         min_act = np.min([np.min(signal[~np.isnan(signal)]) for signal in signals_no_outliers.values() if len(signal) > 0])

#         # plot average signals first to get the x axis labels for the heatmap
#         for j, (name, signal) in enumerate(signals_no_outliers.items()):
#             hm_ax = axs[i*2, j]
#             hm, _ = plot_utils.plot_stacked_heatmap_avg(signal, t, hm_ax, axs[i*2+1, j],
#                                              x_label=x_label, y_label=y_label,
#                                              title='{} {}'.format(key, sub_titles_dict[name]),
#                                              show_cbar=False, cmap=cmap, vmax=max_act, vmin=min_act)

#             # plot any horizontal trial markers
#             if not trial_markers is None:
#                 if name in trial_markers:
#                     for t_idx in trial_markers[name]:
#                         hm_ax.axhline(t_idx - 0.5, dashes=[4, 4], c='k', lw=1)

#         # share y axis for all average rate plots
#         # find largest y range
#         min_y = np.min([ax.get_ylim()[0] for ax in axs[i*2+1, :]])
#         max_y = np.max([ax.get_ylim()[1] for ax in axs[i*2+1, :]])
#         for j in range(axs.shape[1]):
#             axs[i*2+1, j].set_ylim(min_y, max_y)

#         # create color bar legend for heatmap plots
#         fig.colorbar(hm, ax=axs[i*2,:].ravel().tolist(), label=y_label)


def plot_avg_signals(data_mat_dict, parent_groups, parent_group_titles, plot_groups, plot_group_titles, plot_group_labels, t, title, x_label, y_label,
                     parents_on_rows=True, xlims=None, dashlines=None, x_ticks=None, plot_x0=True, legend_params=None, use_se=True,
                     group_colors=None, weights_dict=None, ph=3.5, pw=5):

    def calc_avg(mat, w):
        if w is None:
            return np.nanmean(act, axis=0)
        else:
            return utils.weighted_mean(mat, w, axis=0)

    def calc_error(mat, w):
        if w is None:
            if use_se:
                return utils.stderr(mat, axis=0)
            else:
                return np.std(mat, axis=0)
        else:
            if use_se:
                return utils.weighted_se(mat, w, axis=0)
            else:
                return utils.weighted_std(mat, w, axis=0)

    plotted = False

    if parents_on_rows:
        n_rows = len(parent_groups)
        n_cols = len(plot_groups)
        sharey='row'
    else:
        n_cols = len(parent_groups)
        n_rows = len(plot_groups)
        sharey='col'

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(pw*n_cols, ph*n_rows), layout='constrained', sharey=sharey)

    if n_rows == 1:
        axs = axs[None,:]
    if n_cols == 1:
        axs = axs[:,None]

    fig.suptitle(title)

    for i, (parent_group, parent_group_title) in enumerate(zip(parent_groups, parent_group_titles)):

        if not parent_group in data_mat_dict:
            continue

        if type(t) is dict:
            parent_t = t[parent_group]
        else:
            parent_t = t

        # limit x axis like this so that the y is scaled to what is plotted
        if xlims is None:
            t_sel = np.full(parent_t.shape, True)
        else:
            if type(xlims) is dict:
                t_sel = (parent_t >= xlims[parent_group][0]) & (parent_t <= xlims[parent_group][1])
            else:
                t_sel = (parent_t >= xlims[0]) & (parent_t <= xlims[1])

        for j, (plot_group, plot_group_title) in enumerate(zip(plot_groups, plot_group_titles)):

            if parents_on_rows:
                ax = axs[i,j]
            else:
                ax = axs[j,i]

            ax.set_title(parent_group_title + ' ' + plot_group_title)

            if not dashlines is None:
                plot_utils.plot_dashlines(dashlines, ax=ax)

            with warnings.catch_warnings():
                # runtime warnings are thrown when a slice in the whole act matrix in nan, which can happen
                warnings.simplefilter('ignore', RuntimeWarning)

                for k, group in enumerate(plot_group):
                    if group in data_mat_dict[parent_group]:
                        act = data_mat_dict[parent_group][group]
                        w = None if weights_dict is None else weights_dict[parent_group][group]
                        c = None if group_colors is None else group_colors[k]

                        if len(act) > 0:
                            plotted = True

                            plot_utils.plot_psth(parent_t[t_sel], calc_avg(act, w)[t_sel], calc_error(act, w)[t_sel], ax, label=plot_group_labels[group], color=c, plot_x0=plot_x0)

            if j == 0:
                ax.set_ylabel(y_label)
            else:
                ax.yaxis.set_tick_params(which='both', labelleft=True)
            ax.set_xlabel(x_label)

            if not x_ticks is None:
                ax.set_xticks(x_ticks['ticks'])
                ax.set_xticklabels(x_ticks['labels'])

            if legend_params is False:
                pass
            else:
                if not legend_params is None:
                    if parent_group in legend_params and not legend_params[parent_group] is None:
                        ax.legend(**legend_params[parent_group])
                    else:
                        ax.legend(**legend_params)
                else:
                    ax.legend(loc='best')

    return fig, plotted

def plot_avg_signals_multi_align(data_mat_dict, aligns, parent_groups, parent_group_titles, plot_groups, plot_group_titles,
    plot_group_labels, t, title, x_label, y_label, parents_on_rows=True, xlims=None, dashlines=None,
    x_ticks=None, plot_x0=None, legend_params=None, use_se=True, group_colors=None, weights_dict=None, ph=3.5, pw=5):

    def calc_avg(mat, w):
        if w is None:
            return np.nanmean(act, axis=0)
        else:
            return utils.weighted_mean(mat, w, axis=0)

    def calc_error(mat, w):
        if w is None:
            if use_se:
                return utils.stderr(mat, axis=0)
            else:
                return np.std(mat, axis=0)
        else:
            if use_se:
                return utils.weighted_se(mat, w, axis=0)
            else:
                return utils.weighted_std(mat, w, axis=0)

    plotted = False

    if parents_on_rows:
        n_rows = len(parent_groups)
        n_cols = len(plot_groups)
        sharey='row'
    else:
        n_cols = len(parent_groups)
        n_rows = len(plot_groups)
        sharey='col'

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(pw*n_cols, ph*n_rows), layout='constrained', sharey=sharey)

    if n_rows == 1:
        axs = axs[None, :]
    if n_cols == 1:
        axs = axs[:, None]

    fig.suptitle(title)

    plotted = False

    for i, (parent_group, parent_group_title) in enumerate(zip(parent_groups, parent_group_titles)):

        for j, (align, plot_group, plot_group_title) in enumerate(zip(aligns, plot_groups, plot_group_titles)):

            if parents_on_rows:
                ax = axs[i,j]
            else:
                ax = axs[j,i]

            parent_t = t[align][parent_group]
            dash = None if dashlines is None else dashlines[align]
            colors = None if group_colors is None else group_colors[j]
            xl = None if xlims is None else xlims[align][parent_group]
            x0 = True if plot_x0 is None else plot_x0[align]

            ax.set_title(parent_group_title + ' ' + plot_group_title)

            # limit x axis like this so that the y is scaled to what is plotted
            if xlims is None:
                t_sel = np.full(parent_t.shape, True)
            else:
                t_sel = (parent_t > xl[0]) & (parent_t < xl[1])

            if dash is not None:
                plot_utils.plot_dashlines(dash, ax=ax)

            with warnings.catch_warnings():
                # runtime warnings are thrown when a slice in the whole act matrix in nan, which can happen
                warnings.simplefilter('ignore', RuntimeWarning)

                for k, group in enumerate(plot_group):

                    if group in data_mat_dict[align][parent_group]:
                        act = data_mat_dict[align][parent_group][group]
                        w = None if weights_dict is None else weights_dict[align][parent_group][group]
                        c = None if colors is None else colors[k]

                        if len(act) > 0:
                            plotted = True

                            plot_utils.plot_psth(parent_t[t_sel], calc_avg(act, w)[t_sel], calc_error(act, w)[t_sel], ax, label=plot_group_labels[group], color=c, plot_x0=x0)

            if j == 0:
                ax.set_ylabel(y_label)
            else:
                ax.yaxis.set_tick_params(which='both', labelleft=True)
            ax.set_xlabel(x_label[align])

            if not x_ticks[align] is None:
                ax.set_xticks(x_ticks[align]['ticks'])
                ax.set_xticklabels(x_ticks[align]['labels'])

            if legend_params is False:
                pass
            else:
                if not legend_params is None and not legend_params[align] is None:
                    if parent_group in legend_params[align]:
                        if not legend_params[align][parent_group] is None:
                            ax.legend(**legend_params[align][parent_group])
                        else:
                            ax.legend(loc='best')
                    else:
                        ax.legend(**legend_params[align])
                else:
                    ax.legend(loc='best')

    return fig, plotted


def get_signal_type_labels(signal_type):
    ''' Get signal titles and labels based on the type of signal '''
    match signal_type:
        case 'raw_lig':
            title = 'Raw Ligand Signal'
            ax_label = 'Signal (V)'
        case 'raw_iso':
            title = 'Raw Isosbestic Signal'
            ax_label = 'Signal (V)'
        case 'filtered_lig':
            title = 'Filtered Ligand Signal'
            ax_label = 'Signal (V)'
        case 'filtered_iso':
            title = 'Filtered Isosbestic Signal'
            ax_label = 'Signal (V)'
        case 'baseline_corr_lig':
            title = 'Baseline-Subtracted Ligand Signal'
            ax_label = 'ΔF'
        case 'baseline_corr_iso':
            title = 'Baseline-Subtracted Isosbestic Signal'
            ax_label = 'ΔF'
        case 'fitted_iso':
            title = 'Fitted Isosbestic Signal'
            ax_label = 'Signal (V)'
        case 'fitted_baseline_iso':
            title = 'Fitted Baseline-Subtracted Isosbestic Signal'
            ax_label = 'ΔF'
        case 'dff_iso':
            title = 'Isosbestic Normalized Signal'
            ax_label = '% dF/F'
        case 'z_dff_iso':
            title = 'Z-scored Isosbestic Normalized Signal'
            ax_label = 'Z-dF/F'
        case 'dff_iso_baseline':
            title = 'Baseline-Subtracted Isosbestic Normalized Signal'
            ax_label = '% dF/F (Baseline)'
        case 'z_dff_iso_baseline':
            title = 'Z-scored Baseline-Subtracted Isosbestic Normalized Signal'
            ax_label = 'Z-dF/F (Baseline)'
        case 'dff_iso_fband':
            title = 'Freq-Band Isosbestic Normalized Signal'
            ax_label = '% dF/F (Freq-Band)'
        case 'z_dff_iso_fband':
            title = 'Z-scored Freq-Band Isosbestic Normalized Signal'
            ax_label = 'Z-dF/F (Freq-Band)'
        case 'dff_iso_baseline_fband':
            title = 'Baseline-Subtracted Freq-Band Isosbestic Normalized Signal'
            ax_label = '% dF/F (Baseline & Freq-Band)'
        case 'z_dff_iso_baseline_fband':
            title = 'Z-scored Baseline-Subtracted Freq-Band Isosbestic Normalized Signal'
            ax_label = 'Z-dF/F (Baseline & Freq-Band)'


    return title, ax_label


def get_implant_rel_side(abs_side, implant_side):
    if abs_side is None or abs_side == 'none':
        return None
    else:
        return 'ipsi' if implant_side == abs_side else 'contra'

def get_implant_abs_side(rel_side, implant_side):
    if rel_side is None or rel_side == 'none':
        return None
    else:
        sides = np.array(['left', 'right'])
        return implant_side if rel_side == 'ipsi' else sides[sides != implant_side][0]


def plot_power_spectra(signals, dt, f_max=20, title='', log_x=True, signal_names=None):
    #signal = signal - np.mean(signal)
    # ps = np.abs(np.fft.rfft(signal))**2
    # freqs = np.fft.rfftfreq(signal.size, dt)
    n_sigs = len(signals)

    if signal_names is None:
        signal_names = ['' for i in range(n_sigs)]

    fig, ax = plt.subplots(1)

    if log_x:
        x_lims = [0.001, f_max]
    else:
        x_lims = [0, f_max]

    for i in range(n_sigs):
        signal = signals[i]

        freqs, ps = fp_utils.calc_power_spectra(signal, dt, f_max=f_max)

        freq_sel = (freqs > x_lims[0]) & (freqs < x_lims[1])

        ax.plot(freqs[freq_sel], ps[freq_sel], label=signal_names[i])

    ax.set_yscale('log')
    if log_x:
        ax.set_xscale('log')

    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Power Spectral Density (V^2/Hz)')
    ax.set_title(title)
    ax.legend()

    return fig


def remove_outliers(mat, outlier_thresh):
    mat[np.abs(mat) > outlier_thresh] = np.nan

    return mat


def stack_fp_mats(mat_dict, regions, subj_sess_ids, subjects, signal_type, filter_outliers=False, outlier_thresh=20, groups=None):

    if not utils.is_list(subjects):
        subjects = [subjects]

    stacked_mats = {region: {} for region in regions}
    if groups is None:
        groups = mat_dict[subj_sess_ids[subjects[0]][0]][signal_type][regions[0]].keys()

    for region in regions:
        for group in groups:
            for subj_id in subjects:
                mat = np.vstack([mat_dict[sess_id][signal_type][region][group] for sess_id in subj_sess_ids[subj_id] if group in mat_dict[sess_id][signal_type][region].keys()])
                if filter_outliers:
                    mat = remove_outliers(mat, outlier_thresh)

                # TODO: This does not work for multiple subjects
                stacked_mats[region][group] = mat

    return stacked_mats

def sort_subj_regions(subj_regions, all_regions):
    plot_order = {r: i for i, r in enumerate(all_regions)}
    return sorted(subj_regions, key=lambda x: next((plot_order[r] for r in all_regions if r in x), np.inf))

def get_region_idx_mapping(subj_regions, all_regions):
    return np.array([next(i for i, r in enumerate(all_regions) if r in sr) for sr in subj_regions])

def count_corr_pairs(subj_regions, all_regions):

    all_region_idx_mapping = get_region_idx_mapping(subj_regions, all_regions)

    # check for duplicate region indices
    idx_counts = Counter(all_region_idx_mapping)

    pair_counts = {}

    # same region auto-correlations
    for r, n in idx_counts.items():
        pair_counts[(r, r)] = n

    # cross region cross-correlations
    for r1, r2 in itertools.combinations(idx_counts.keys(), 2):
        key = tuple(sorted([r1,r2]))
        pair_counts[key] = idx_counts[r1] * idx_counts[r2]

    return pair_counts


# %% Alignment Helpers

class Alignment(StrEnum):
    cport_on = 'cport_on'
    cpoke_in = 'cpoke_in'
    early_cpoke_in = 'early_cpoke_in'
    tone = 'tone'
    cue = 'cue'
    cpoke_out = 'cpoke_out'
    early_cpoke_out = 'early_cpoke_out'
    resp = 'resp'
    reward = 'reward'
    cue_poke_resp = 'norm_cue_poke_resp'
    poke_cue_resp = 'norm_poke_cue_resp'
    resp_reward = 'norm_resp_reward'
    cue_resp = 'norm_cue_resp'
    cue_reward = 'norm_cue_reward'
    cport_on_cpoke_in = 'norm_cport_on_cpoke'

def get_align_title(align):
    match align:
        case Alignment.cport_on:
            return 'Center Port On'
        case Alignment.cpoke_in:
            return 'Center Poke In'
        case Alignment.early_cpoke_in:
            return 'Early Center Poke In'
        case Alignment.tone:
            return 'Tone Start'
        case Alignment.cue:
            return 'Response Cue'
        case Alignment.cpoke_out:
            return 'Center Poke Out'
        case Alignment.early_cpoke_out:
            return 'Early Center Poke Out'
        case Alignment.resp:
            return 'Response Poke'
        case Alignment.reward:
            return 'Reward Delivery'
        case Alignment.cue_poke_resp:
            return 'Normalized Cue, Poke Out, & Response'
        case Alignment.poke_cue_resp:
            return 'Normalized Poke Out, Cue, & Response'
        case Alignment.resp_reward:
            return 'Normalized Response to Reward'
        case Alignment.cue_resp:
            return 'Normalized Cue to Response'
        case Alignment.cue_reward:
            return 'Normalized Cue, Poke Out, Response & Reward'
        case Alignment.cport_on_cpoke_in:
            return 'Normalized Center Port On to Poke In'
        case _:
            return align

def get_align_xlabel(align):
    gen_x_label = 'Time from {} (s)'
    gen_norm_x_label = 'Normalized Time from {}'
    match align:
        case Alignment.cport_on:
            return gen_x_label.format('port on')
        case Alignment.cpoke_in:
            return gen_x_label.format('poke in')
        case Alignment.early_cpoke_in:
            return gen_x_label.format('poke in')
        case Alignment.tone:
            return gen_x_label.format('tone onset')
        case Alignment.cue:
            return gen_x_label.format('response cue')
        case Alignment.cpoke_out:
            return gen_x_label.format('poke out')
        case Alignment.early_cpoke_out:
            return gen_x_label.format('poke out')
        case Alignment.resp:
            return gen_x_label.format('response poke')
        case Alignment.reward:
            return gen_x_label.format('reward')
        case Alignment.cue_poke_resp:
            return gen_norm_x_label.format('response cue to response')
        case Alignment.poke_cue_resp:
            return gen_norm_x_label.format('poke out to response')
        case Alignment.resp_reward:
            return gen_norm_x_label.format('response to reward')
        case Alignment.cue_resp:
            return gen_norm_x_label.format('cue to response')
        case Alignment.cue_reward:
            return gen_norm_x_label.format('cue to reward')
        case Alignment.cport_on_cpoke_in:
            return gen_norm_x_label.format('center port on to poke')


# %% Plot Image Saving/Loading Methods

def save_fig(fig, save_path, format='png', **kwargs):

    utils.check_make_dir(save_path)

    if not format in save_path:
        save_path += '.' + format

    if format == 'svg':
        plt.rcParams["svg.fonttype"] = 'none'

    fig.savefig(save_path, **kwargs)


def get_base_figure_save_path():
    # make sure subject id is a string
    return path.join(utils.get_user_home(), 'FP Images')


def get_figure_save_path(behavior, subj_ids, filename=''):
    # make sure subject id is a string
    subj_ids = str(subj_ids) if not utils.is_list(subj_ids) else '_'.join([str(i) for i in sorted(subj_ids)])
    return path.join(get_base_figure_save_path(), behavior, subj_ids, filename)


def generate_figure_ppt(save_path, group_by=None, behaviors=None, subjects=None, alignments=None, filenames=None, signal_types=None):
    '''
    Generate a power point from saved figures for the specified behaviors, subjects, and alignments grouped by the provided ordering

    Parameters
    ----------
    save_path : The path to the saved powerpoint
    group_by : The grouping order for the images. The default is ['behavior', 'subject', 'alignment', 'filename'].
        This determines the order in which images are placed in the powerpoint
    behaviors : The behaviors to include in the powerpoint. The default is all behaviors.
    subjects : The subject ids to include in the powerpoint. The default is all subjects.
    alignments : The alignment points to include in the powerpoint. The default is all alignments.

    Returns
    -------
    None.

    '''

    base_path = get_base_figure_save_path()
    def_group_by = ['behavior', 'subject', 'alignment', 'filename']
    if group_by is None:
        group_by = def_group_by
    elif not utils.is_list(group_by):
        group_by = [group_by]

    if len(group_by) < len(def_group_by):
        group_by.extend([g for g in def_group_by if g not in group_by])

    if behaviors is None:
        behaviors = [f.name for f in os.scandir(base_path) if f.is_dir()]
    elif not utils.is_list(behaviors):
        # make sure a single behavior is in list form
        behaviors = [behaviors]

    if subjects is None:
        subj_id_paths = utils.flatten([glob(path.join(base_path, beh, '*')) for beh in behaviors])
        subjects = np.unique([path.basename(p) for p in subj_id_paths]).tolist()
    elif not utils.is_list(subjects):
        subjects = [subjects]

    # make sure subject ids are strings and collapse multiple ids into
    subjects = [str(i) if not utils.is_list(i) else '_'.join([str(j) for j in sorted(i)]) for i in subjects]

    if alignments is None:
        alignments = [a for a in Alignment]
    elif not utils.is_list(alignments):
        # make sure a single alignment is in list form
        alignments = [alignments]

    if filenames is None:
        # get all possible filenames of figures
        all_filenames = utils.flatten([glob(path.join(base_path, b, s, a+'*')) for b in behaviors for s in subjects for a in alignments])
        # remove path and extension
        filenames = np.unique([path.basename(f).replace('.png', '') for f in all_filenames])
        # remove alignment
        filenames = [f.replace(a+'_', '', 1) for a in alignments for f in filenames if a in f]
    elif not utils.is_list(filenames):
        # make sure a single filename is in list form
        filenames = [filenames]

    # make sure there are no duplicates and the order is preserved if passed in
    _, idx = np.unique(filenames, return_index=True)
    filenames = np.array(filenames)[np.sort(idx)]

    if not signal_types is None:
        # remove any filenames that do not have the given signal types in their name
        if not utils.is_list(signal_types):
            signal_types = [signal_types]

        filenames = [f for f in filenames if any([t in f for t in signal_types])]

    prs = Presentation()
    prs.slide_width = Inches(16)
    prs.slide_height = Inches(9)

    __add_ppt_slides_recursive(prs, group_by, behaviors, subjects, alignments, filenames, 0)

    if not '.pptx' in save_path:
        save_path += '.pptx'
    prs.save(save_path)


def __add_ppt_slides_recursive(prs, group_by, behaviors, subjects, alignments, filenames, level, level_titles=[]):

    group = group_by[level]
    title_font_size = Pt(60 - 10*level)

    match group:
        case 'behavior':
            level_groups = behaviors
            level_group_labels = behaviors
        case 'subject':
            level_groups = subjects
            level_group_labels = [s.replace('_', ', ') for s in subjects]
        case 'alignment':
            level_groups = alignments
            level_group_labels = [get_align_title(a) for a in alignments]
        case 'filename':
            level_groups = filenames
            level_group_labels = filenames

    # if we haven't reached the lowest level make a title slide for each level group and recurse
    if level < len(group_by)-1:
        for i, level_group in enumerate(level_groups):

            match group:
                case 'behavior':
                    images_exist = __check_images_exist(level_group, subjects, alignments, filenames)
                    need_title = __check_images_exist(level_group, subjects, alignments, filenames, count_thresh=1)
                case 'subject':
                    images_exist = __check_images_exist(behaviors, level_group, alignments, filenames)
                    need_title = __check_images_exist(behaviors, level_group, alignments, filenames, count_thresh=1)
                case 'alignment':
                    images_exist = __check_images_exist(behaviors, subjects, level_group, filenames)
                    need_title = __check_images_exist(behaviors, subjects, level_group, filenames, count_thresh=1)
                case 'filename':
                    images_exist = __check_images_exist(behaviors, subjects, alignments, level_group)
                    need_title = __check_images_exist(behaviors, subjects, alignments, level_group, count_thresh=1)

            if images_exist:
                level_titles_copy = level_titles.copy()
                level_titles_copy.append(level_group_labels[i])

                if need_title:
                    layout = prs.slide_layouts[0] # title slide
                    slide = prs.slides.add_slide(layout)
                    title = slide.shapes.title
                    title.text = '\n'.join(level_titles_copy)
                    for line in title.text_frame.paragraphs:
                        line.font.size = title_font_size
                        if level == 0:
                            line.font.bold = True

                    # need to reset the height and top after changing width and left
                    title_height = title.height
                    title_top = title.top
                    title.width = prs.slide_width
                    title.left = 0
                    title.height = title_height
                    title.top = title_top

                match group:
                    case 'behavior':
                        __add_ppt_slides_recursive(prs, group_by, level_group, subjects, alignments, filenames, level+1, level_titles_copy)
                    case 'subject':
                        __add_ppt_slides_recursive(prs, group_by, behaviors, level_group, alignments, filenames, level+1, level_titles_copy)
                    case 'alignment':
                        __add_ppt_slides_recursive(prs, group_by, behaviors, subjects, level_group, filenames, level+1, level_titles_copy)
                    case 'filename':
                        __add_ppt_slides_recursive(prs, group_by, behaviors, subjects, alignments, level_group, level+1, level_titles_copy)

    else:
        # we are at the lowest level so we need to start adding slides
        for i, level_group in enumerate(level_groups):
            # get all images the for the given groups
            # Note: at the lowest level there will be one element for each of the grouping arguments
            if group == 'alignment':
                if len(level_group) != 0 and not alignments.endswith('_'):
                    level_group += '_'
            else:
                if len(alignments) != 0 and not alignments.endswith('_'):
                    alignments += '_'

            match group:
                case 'behavior':
                    image_files = glob(get_figure_save_path(level_group, subjects, alignments+filenames+'.*'))
                case 'subject':
                    image_files = glob(get_figure_save_path(behaviors, level_group, alignments+filenames+'.*'))
                case 'alignment':
                    image_files = glob(get_figure_save_path(behaviors, subjects, level_group+filenames+'.*'))
                case 'filename':
                    image_files = glob(get_figure_save_path(behaviors, subjects, alignments+level_group+'.*'))

            image_files = sorted(image_files)
            level_titles_copy = level_titles.copy()
            level_titles_copy.append(level_group_labels[i])

            if len(image_files) > 1:
                layout = prs.slide_layouts[0] # title slide
                slide = prs.slides.add_slide(layout)
                title = slide.shapes.title
                title.text = '\n'.join(level_titles_copy)
                for line in title.text_frame.paragraphs:
                    line.font.size = title_font_size

                # need to reset the height and top after changing width and left
                title_height = title.height
                title_top = title.top
                title.width = prs.slide_width
                title.left = 0
                title.height = title_height
                title.top = title_top

            for image_file in image_files:

                layout = prs.slide_layouts[5] # title only
                slide = prs.slides.add_slide(layout)
                title = slide.shapes.title
                title.text = ', '.join(level_titles_copy)
                title.text_frame.paragraphs[0].font.size = title_font_size
                # need to reset the height and top after changing width and left
                title_height = title.height
                title.width = prs.slide_width
                title.height = title_height
                title.left = 0
                title.top = 0

                image = slide.shapes.add_picture(image_file, 0, 0, width=prs.slide_width - Inches(0.1))

                # rescale image if too tall
                if image.height > prs.slide_height - title.height:
                    aspect = image.height/image.width
                    image.height = prs.slide_height - title.height - Inches(0.1)
                    image.width = int(image.height/aspect)

                image.left = int((prs.slide_width - image.width) / 2)
                image.top = int((prs.slide_height - image.height - title.height) / 2) + title.height

def __check_images_exist(behaviors, subjects, alignments, filenames, count_thresh = 0):

    if not utils.is_list(behaviors):
        behaviors = [behaviors]

    if not utils.is_list(subjects):
        subjects = [subjects]

    if not utils.is_list(alignments):
        alignments = [alignments]

    if not utils.is_list(filenames):
        filenames = [filenames]

    file_count = 0

    for b in behaviors:
        for s in subjects:
            for a in alignments:
                if len(a) != 0 and not a.endswith('_'):
                    a += '_'

                for f in filenames:
                    files = glob(path.join(get_base_figure_save_path(), b, s, a+f+'.*'))
                    if len(files) > 0:
                        file_count += 1
                    if file_count > count_thresh:
                        return True
                if file_count > count_thresh:
                    return True
            if file_count > count_thresh:
                return True
        if file_count > count_thresh:
            return True

    return False
