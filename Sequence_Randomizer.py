import string
import os
import sys
from random import shuffle
# Script that reads and save a randomized sequence of the events found in the
# input folder. The events are divided in the same number of files based on
# in the given output folder.

IDir = sys.argv[1]
ODir = sys.argv[2]

fnames = os.listdir(IDir)

events_list = []
for fn in fnames:
    f = open(IDir + fn, 'r')
    for l in f:
        v = l.strip().split()
        events_list.append([v[i] for i in range(len(v))])
    f.close()

TOT_EVENTS = len(events_list)
EVENTS_PER_FILE = TOT_EVENTS/len(fnames) + 1

for i in range(3):
    shuffle(events_list)

File_Count = 0
Events_Count = 0

for i, e in enumerate(events_list):
    if Events_Count == 0:
        of = open(ODir + 'random_%04d.dat' % File_Count, 'w')
        File_Count += 1

    for ev_id in e[:-1]:
        of.write('%s\t' % ev_id)
    of.write('%s\n' % e[-1])
    Events_Count += 1

    if Events_Count == EVENTS_PER_FILE or i == len(events_list):
        of.close()
        Events_Count = 0
