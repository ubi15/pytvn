import os, sys

import numpy as np
import matplotlib.pyplot as plt

IDir = sys.argv[1]
ODir = sys.argv[2]


Files = sorted(os.listdir(IDir))

nFiles = len(Files)


avg_collabs = np.zeros(nFiles)
eve_time = np.zeros(nFiles)

for Find, fn in enumerate(Files):
    NPap = 0
    NCol = 0
    with open(os.path.join(IDir, fn), 'rb') as f:
        for l in f:
            NPap += 1
            NCol += len(l.strip().split()) - 1

    if NPap > 0:
        avg_collabs[Find] = float(NCol) / float(NPap)

    if Find == 0:
        eve_time[Find] = NPap
    else:
        eve_time[Find] = eve_time[Find-1] + NPap


with open(os.path.join(ODir, "00/rhos/zzz_AvgColl.dat"), "wb") as of:
    for i in range(nFiles):
        of.write("%d\t%d\t%.03e\n" % (i, eve_time[i], avg_collabs[i]))

plt.loglog(range(1, nFiles+1), avg_collabs, '-k', label=r"$<Coll(months)>$")
plt.loglog(eve_time, avg_collabs, '-r', label=r"$<Coll(events)>$")
plt.savefig(os.path.join(ODir, "00/rhos/zzz_AvgColl.pdf"))



