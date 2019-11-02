import numpy as np
from matplotlib import pyplot as plt


Stats = {}

Stats['TWT'] = {}
Stats['YHO'] = {}

Stats['TWT']['Total'] = {}
Stats['YHO']['Total'] = {}

Stats['TWT']['Total']['Events'] = 1.638e+7
Stats['YHO']['Total']['Events'] = 3.180e+6

Stats['TWT']['Total']['Edges'] = 3.793e+6
Stats['YHO']['Total']['Edges'] = 9.052e+5

Stats['TWT']['Total']['Nodes'] = 5.378e+5
Stats['YHO']['Total']['Nodes'] = 1.000e+5

Stats['TWT']['Soc01'] = {}
Stats['YHO']['Soc01'] = {}

Stats['TWT']['Soc01']['Events'] = 1.383e+7
Stats['YHO']['Soc01']['Events'] = 2.745e+6

Stats['TWT']['Soc01']['Edges'] = 2.320e+6
Stats['YHO']['Soc01']['Edges'] = 6.274e+5

Stats['TWT']['Soc01']['Nodes'] = 3.203e+5
Stats['YHO']['Soc01']['Nodes'] = 9.126e+4

Stats['TWT']['Soc02'] = {}
Stats['YHO']['Soc02'] = {}

Stats['TWT']['Soc02']['Events'] = 1.212e+7
Stats['YHO']['Soc02']['Events'] = 2.360e+6

Stats['TWT']['Soc02']['Edges'] = 1.275e+6
Stats['YHO']['Soc02']['Edges'] = 3.368e+5

Stats['TWT']['Soc02']['Nodes'] = 2.304e+5
Stats['YHO']['Soc02']['Nodes'] = 7.438e+4


for k in Stats.keys():
    for j in Stats[k].keys():
        print '%s\t%s\t%r' % (k, j, Stats[k][j])

