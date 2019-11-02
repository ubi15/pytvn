import string, os, sys, gzip, cPickle
from numpy import ceil

'''
Usage:
    python2 Inter_Event_TWT_like.py IDIR ODIR

Python script reading the data from files within IDIR divided per
day/hor/whatever and writing two files in ODIR:
    - Interevent_DIST.dat =>
        interevent_val \t freq_index_intrevent_tot \t freq_index_intrevent_clr \n

    - Interevent_AVG.dat =>
        average_inter_tot \t average_inter_clr \n
'''


IDIR = sys.argv[1]
ODIR = sys.argv[2]


fnames = sorted(os.listdir(IDIR))

PerFile = 24 # in hours...

Moment = 1.  # The exponent of the inter event moment for each node.

Nodes = {}
Inter_Clr = [0]*PerFile*len(fnames)
Inter_Tot = [0]*PerFile*len(fnames)

for day, fn in enumerate(fnames):
    sys.stdout.write("Doing file %03d of %03d - %s \r" %\
        (day+1,len(fnames), fn))
    sys.stdout.flush()

    # Loading events..
    Listone = []
    with open(os.path.join(IDIR, fn), 'r') as f:
        for l in f:
            v = l.strip().split()
            Listone.append([v[0], v[1]])

    CLRs = {} # The number of events as caller and as calleds
    Both = {}
    for clr,cld in Listone:
        CLRs.setdefault(clr, 0)
        CLRs[clr] += 1

        Both.setdefault(clr, 0)
        Both[clr] += 1
        Both.setdefault(cld, 0)
        Both[cld] += 1

    for clr, noc in CLRs.items():
        if (clr in Nodes) and (Nodes[clr]["lc"] is not None): # I already called
            dist = PerFile*(day - Nodes[clr]["lc"])
            Inter_Clr[int(ceil(dist))] += 1
            Nodes[clr]["sc"] += float(dist)**Moment
            Nodes[clr]["nc"] += 1.
        else:
            # l: last event, lc: last call,
            # st: sum total inter-event moment, nt: number of total event,
            # sc: sum of caller moment, nc: number of caller moment.
            Nodes.setdefault(clr, {"l": None, "lc": None,\
                    "st": .0, "nt": .0, "sc": .0, "nc": .0})

        if noc > 1:
            dist = PerFile/(noc-1)
            Inter_Clr[int(ceil(dist))] += noc - 1

            Nodes[clr]["sc"] += (noc-1.) * (float(PerFile) / (noc-1.))**Moment
            Nodes[clr]["nc"] += noc-1.

        Nodes[clr]["lc"] = day + .5 # Put the last call in the middle of the day


    for clr, noc in Both.items():
        if (clr in Nodes) and (Nodes[clr]["l"] is not None): # already done
                                                             # at least 1 event
            dist = PerFile*(day - Nodes[clr]["l"])
            Inter_Tot[int(ceil(dist))] += 1
            Nodes[clr]["st"] += float(dist)**Moment
            Nodes[clr]["nt"] += 1.

        else: # first call
            Nodes.setdefault(clr, {"l": None, "lc": None,\
                    "st": .0, "nt": .0, "sc": .0, "nc": .0})

        if noc > 1:
            dist = PerFile/(noc-1)
            Inter_Tot[int(ceil(dist))] += noc - 1
            Nodes[clr]["st"] += (noc-1.) * (float(PerFile) / (noc-1.))**Moment
            Nodes[clr]["nt"] += noc-1.

        Nodes[clr]["l"] = day + .5

with open(os.path.join(ODIR, "Interevent_DIST.dat"), "wb") as f:
    for h,[bt,bc] in enumerate(zip(Inter_Tot, Inter_Clr)):
        f.write("%d\t%e\t%e\n" % (h,bt,bc))

with open(os.path.join(ODIR, "Interevent_AVG.dat"), "wb") as f:
    for k,v in Nodes.items():
        f.write("%e\t%e\n" % (v["st"]/max(1.,v["nt"]), v["sc"]/max(1.,v["nc"])))


