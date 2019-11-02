import sys
import os
import netpython.pynet as PYN
import random as RNDM
import string
import cPickle
import numpy as np
import gzip
import copy as copy

def Lin_Log_Bins(bins):
    cc_temp = np.array(bins)
    cc = [np.floor(cc_temp[0]) - .5, np.ceil(cc_temp[1]) + .5]
    for i in cc_temp[]:
        if np.ceil(i) > cc[-1]:
            cc.append(np.ceil(i) + .5)
    cc = np.array(cc)
    return cc


# Checking the number of arguments...
if len(sys.argv) < 5:
    print 'Usage:'
    print 'Network_Socializer IDir ODir SocTHr #times'
    print ''
    print 'Then, optional:'
    print 'initial_time final_time'
    #print 'Error, please provide input dir and output dir!'
    exit(5)

# Gzipped input?
if False:
    ApriFile = gzip.open
else:
    ApriFile = open

# Input and output directories...
IDir = str(sys.argv[1])
ODir = str(sys.argv[2])

# The social threshold (i.e. we keep only edges that have at least this
# weight in both the directions...)
SOC_THR = float(sys.argv[3])

# The number of time values to analyze...
n_t_smpl = int(sys.argv[4])

Social = False
if SOC_THR != .0:
    Social = True

# The list of file names to process...
fnames = os.listdir(IDir)
fnames = sorted(fnames)

# The initial and final time (default 0 -> number of files to be analyzed...)...
tini = 0
tfin = len(fnames)

# If we specify a different slice here we go...
if len(sys.argv) == 6:
    tini = int(sys.argv[5])
if len(sys.argv) == 7:
    tini = int(sys.argv[5])
    tfin = int(sys.argv[6])

# Then we select only the right slice of files to be analyzed...
fnames = fnames[tini:tfin]

# We then initialize the LABEL that will be given to all the output files...
period = '%03d-%03d' % (tini, tfin)
LABEL = '_p-%s_thres_%02d' % (period, SOC_THR)

# Here we check the output structure in the output folder...
if ODir[-1] != '/':
    ODir += '/'

if not os.path.exists(ODir):
    os.mkdir('%s' % ODir)

if not os.path.exists('%s%02d' % (ODir, SOC_THR)):
    os.mkdir('%s%02d' % (ODir, SOC_THR))

ODir += '%02d/' % SOC_THR

if not os.path.exists('%sdata/' % ODir):
    os.mkdir('%sdata' % ODir)

# The growing directed graph...
DG = PYN.Net()

# Save the network Statistic...
Network_Stats = {}

# The auxiliary dictionary to keep track of the activity bin, degree-bin
# and company/non-company of each user...
Act_Com_Dic = {}

# For over the files...
fnames = sorted(fnames)

EV_TOT = 0

# Self explaining (the files store in each row: "CallerID CalledID company_cr company_cd \n")...
for fn in fnames:
    LL = []
    of = ApriFile(os.path.join(IDir, fn), 'rb')
    for ln in of:
        vvvv = ln.strip().split()
        LL.append(vvvv)
    of.close()
    for n12 in LL:
        if n12[0] != n12[1]:
            EV_TOT += 1
            #DG.add_edge(n12[0], n12[1])
            DG[n12[0]][n12[1]] += 1.
            #DG[n12[0]][n12[1]]['weight'] += 1.

            Act_Com_Dic.setdefault(n12[0], {})
            Act_Com_Dic[n12[0]].setdefault('company', bool(int(n12[2])))

            Act_Com_Dic.setdefault(n12[1], {})
            Act_Com_Dic[n12[1]].setdefault('company', bool(int(n12[3])))

# Now evaluating the number of edges...
TOT_NUM_EDGES = .0
TOT_COMPANY_USERS = 0
for n in DG:
    if Act_Com_Dic[n]['company']:
        TOT_COMPANY_USERS += 1

    for n1 in DG[n]:
        if DG[n][n1] and DG[n1][n]:
            TOT_NUM_EDGES += .5
        elif DG[n][n1] and ( not DG[n1][n] ):
            TOT_NUM_EDGES += 1.

print ''
print ''
print 'Total Network:'
print ''
print 'Total Nodes: %1.03e' % len(DG)
print 'Company Nodes: %1.03e' % TOT_COMPANY_USERS
print 'Edges: %1.03e' % TOT_NUM_EDGES
print 'Events: %1.03e' % EV_TOT

Network_Stats['TOT'] = {}
Network_Stats['TOT']['Nodes'] = len(DG)
Network_Stats['TOT']['CompanyNodes'] = TOT_COMPANY_USERS
Network_Stats['TOT']['Edges'] = TOT_NUM_EDGES
Network_Stats['TOT']['Events'] = int(EV_TOT)

# Now, check on the social threshold:
#   - if we set it != 0 we keep only the edges with weight >= SOC_THR;
#   - otherwise we keep the network as is...

if Social:
    # Now we clean the unsocial edges and links...
    for n0 in DG:
        #vicini = DG.neighbors(n0)
        for n1 in DG[n0]:
            if not DG[n1][n0]:
                DG[n0][n1] = .0
            elif (DG[n0][n1] < SOC_THR) or (DG[n1][n0] < SOC_THR):
                DG[n0][n1] = .0
                DG[n1][n0] = .0

    # We now store the orphan nodes (with no neighbors) and we later delete them...
    blacklist = []
    for n0 in DG:
        if DG[n0].deg() == 0:
            blacklist.append(n0)

    #fout = gzip.open(ODir + 'Deleted_' + LABEL + '.dat', 'w')
    for bl in blacklist:
        DG.delNode(bl)
        #print >> fout, bl
    #fout.close()

# A new cycle over the Network to count the true activity based only on the events involving
# only actual social nodes...
EV_TOT = 0
TOT_NUM_EDGES = .0
MAX_ACT = 2.
MIN_ACT = 1000000.
SOC_COMPANY_USERS = 0

for n in DG:
    NODE_ACT = .0

    if Act_Com_Dic[n]['company']:
        SOC_COMPANY_USERS += 1

    for n1 in DG[n].iterOut():
        NODE_ACT += DG[n][n1]
        if DG[n][n1] and DG[n1][n]:
            TOT_NUM_EDGES += .5
        elif DG[n][n1] and ( not DG[n1][n] ):
            TOT_NUM_EDGES += 1.

    EV_TOT += int(NODE_ACT)

    if NODE_ACT > MAX_ACT:
        MAX_ACT = NODE_ACT
    if NODE_ACT < MIN_ACT and NODE_ACT > .0:
        MIN_ACT = NODE_ACT

    Act_Com_Dic[n]['aa_bin'] = NODE_ACT
    Act_Com_Dic[n]['kk_bin'] = .0

Network_Stats['SOC'] = {}
Network_Stats['SOC']['Threshold'] = SOC_THR
Network_Stats['SOC']['Nodes'] = len(DG)
Network_Stats['SOC']['CompanyNodes'] = SOC_COMPANY_USERS
Network_Stats['SOC']['Edges'] = TOT_NUM_EDGES
Network_Stats['SOC']['Events'] = int(EV_TOT)

print ''
print 'Social Network:'
print ''
print 'Threshold: %02d' % SOC_THR
print 'Nodes: %1.03e' % len(DG)
print 'Company Nodes: %1.03e' % SOC_COMPANY_USERS
print 'Edges: %1.03e' % TOT_NUM_EDGES
print 'Events: %1.03e' % EV_TOT


# Now that we created the graph let us sort the keys by their out
# activity value so that we can compute the p(n,a)
# Binning the activities and their neighborhood...
# First of all collect the activities, keep only the a > 0 and then bin them with
# logarithmically spaced bins...
aa = np.array([Act_Com_Dic[peer]['aa_bin']\
        for peer in DG if Act_Com_Dic[peer]['company']])
aa = aa[aa != .0]
n_a_bins = int(np.floor(np.log10(aa.max()/aa.min())/np.log10(1.5))) + 1
aa_freq, aa_bins = np.histogram(aa,\
        np.logspace(np.log10(aa.min()*.999), np.log10(aa.max()*1.001), n_a_bins))

print 'DEBUG n_act_bins = %02d' % n_a_bins
# Updating label...
LABEL = '_p-%s_thres_%02d_a%02d' % (period, SOC_THR, n_a_bins)

# We now initialize the dictionary containing all the bins...
Bins = {}
Bins['aa'] = copy.deepcopy(aa_bins)
Bins['kk'] = {}

# Now we sort the nodes depending on their activity value...
for N in DG:
    if Act_Com_Dic[N]['company']:
        i = 0
        if Act_Com_Dic[N]['aa_bin'] > .0:
            while not( (Act_Com_Dic[N]['aa_bin'] >= aa_bins[i]) and (Act_Com_Dic[N]['aa_bin'] < aa_bins[i+1])):
                i += 1
            Act_Com_Dic[N]['aa_bin'] = i
        else:
            Act_Com_Dic[N]['aa_bin'] = 0

# Now we save the actual activity of the nodes in the bins...
aa /= aa.max()
aa_freq, aa_bins = np.histogram(aa,\
        np.logspace(np.log10(aa.min()*.999), np.log10(aa.max()*1.001), n_a_bins))

# Now we sort the nodes within a single activity class depending on their final degree...

# For over the activity class...
for i in range(n_a_bins):
    # First we sort the degree for the current activity class (practically the P(a,k,t_fin))...
    nodi = [n for n in DG if ((Act_Com_Dic[n]['company']) and (Act_Com_Dic[n]['aa_bin'] == i))]
    if nodi:
        # The final degree...
        deg_fin = [float(DG[n].deg()) for n in nodi]


        # Then we split the nodes in n_bins subclasses...
        n_k_bins = max(2,\
                int(np.floor(np.log10(max(deg_fin)/min(deg_fin))/np.log10(1.5))) + 1)

        print 'DEBUG: in act class %02d   ---   nkbins %02d' % (i, n_k_bins)
        print 'DEBUG:                     ---   nodes %d , kmin=%d, kmax=%d'\
                % (len(deg_fin), min(deg_fin), max(deg_fin))

        k_bins = np.logspace(np.log10(max(1., min(deg_fin)*.999)),\
                np.log10(max(deg_fin)*1.001), n_k_bins)
        # Saving the degree bins for this activity class
        Bins['kk'][i] = copy.deepcopy(k_bins)

        for jj, nn in enumerate(nodi):
            bb = 0
            if deg_fin[jj]:
                while not( (deg_fin[jj] >= k_bins[bb]) and (deg_fin[jj] < k_bins[bb + 1]) ):
                    bb += 1
            else:
                bb = 0
            Act_Com_Dic[nn]['kk_bin'] =  bb

# Now we add the binning of the nodes based solely on their out-degree, regardless of the
# activity...
kk_alone = np.array([float(DG[peer].deg())\
        for peer in DG if Act_Com_Dic[peer]['company']])
kk_alone = kk_alone[kk_alone != .0]
log2_min = np.floor(np.log2(kk_alone.min()))
log2_max = np.ceil(np.log2(kk_alone.max()))

# The vector containing the 2-based logarithmically spaced bins for the degree alone sorting
# and the population counter...
kk_binning_bins = np.array([2.**expi for expi in range(int(log2_min), int(log2_max) + 1)])
Bins['kk_alone'] = copy.deepcopy(kk_binning_bins)
N_K = {}
# Now we sort the nodes depending on their degree value...
for N in DG:
    if Act_Com_Dic[N]['company']:
        deg_fin = np.floor(np.log2(float(DG[N].deg())))
        if deg_fin >= .0:
            Act_Com_Dic[N].setdefault('degree_alone_bin', deg_fin)
            N_K.setdefault(int(deg_fin), .0)
            N_K[int(deg_fin)] += 1.
        else:
            Act_Com_Dic[N].setdefault('degree_alone_bin', -1.)
            N_K.setdefault(-1, .0)
            N_K[-1] += 1.


# Now a last cycle to initialize the following variables attached to each node and edge:
# - for each edge a boolean that will keep track of the so far activated edges;
# - for each node a boolean flag to keep track of the node's state (activated/non activated);
# - and a float to keep track of the node's degree;
# They will let us use only one variable in the importer run.
# We count the population in each bin and sub bin...
N_A_K = {}
for node in DG:
    if Act_Com_Dic[node]['company']:
        act_bin = Act_Com_Dic[node]['aa_bin']
        deg_bin = Act_Com_Dic[node]['kk_bin']
        N_A_K.setdefault(act_bin, {})
        N_A_K[act_bin].setdefault(deg_bin, 0)
        N_A_K[act_bin][deg_bin] += 1
    Act_Com_Dic[node]['nshots'] = .0


# The label for all the file saved from here on...
LABEL = LABEL + '_step-by-step_t-%02d' % n_t_smpl

################################################################

# The dictionaries that will take care of counting:
# how many nodes pass at degree k...
Njumps = {}

# how long they stay over that k value...
Nshots = {}

# The single stories...
pna_singles = {}

# New, unique dictionary that stores all the p_n_related stuff in one
# structure only, i.e., we don't have to save two different dictionary
# structures to save two vectors...
# The structure is as follows:
# P_N_A[ak][dk][n] = {'s_avg': .0, 's_wal': .0, 's_new': .0, 's_eve': .0}
# where:
# - 's_avg' is the sum of the new_eve/num_shots for that n counting also
#   the 0/shots (change of degree for external activity);
# - 's_wal' is the number of walker actually shooting at the degree n
#   (if a node jumps at n and then moves to degree n+1 without ever being
#   active we don't count it);
# - 's_new' is the counter of events toward new nodes by nodes at degree n;
# - 's_eve' is the counter of the total events done by any node at degree n;
#
# Given these variables we compute the p_n_average as:
#
# p_n_av(n, ak, dk) = s_avg/s_wal,
#
# while the p_n_events will be defined as:
#
# p_n_ev(n, ak, dk) = s_new/s_eve
#
P_N_A = {}

# ... and the same for the degree-alone binning...
P_N_K = {}

# Time list...
Net_Time = []

# The number of active nodes per time...
N_active_t = []

# The number of active edges per time...
E_active_t = []

# And the number of events so far...
N_events_t = []

# Plus the average degree up to time t for the whole network = 2E/N
K_events_t = []

# The dictionary that contains the P(a, k, t)
Pakt = {}

# and the ones containing the k(a, t) and n(a, t) (number of active nodes in that
# bin at that time...), so that <k(a, t)> = k(a, t)/n(a, t)...
k_a_t = {}
n_a_t = {}

# Generate the times to analyze for the P(a,k,t)...
TVec = np.ceil(np.logspace(np.log10(5.), np.log10(float(len(fnames))), n_t_smpl))
i = 0
while i < len(TVec):
    if i < (len(TVec) - 1) and TVec[i] == TVec[i + 1]:
        TVec = np.delete(TVec, i)
    else:
        i += 1

# And check that the last one is correct...
if TVec[-1] != float(len(fnames)):
    TVec[-1] = float(len(fnames))

# DEBUG
print TVec

# Time and events counters...
Time = .0
Events = .0

# The growing number of edges...
TOT_NUM_EDGES = .0

# Reset the network...
# WARNING, SO FAR WE DEVELOPED ONLY THE NON SOCIAL CASE!!!
DG = PYN.Net()

# Main cycle over files...
for ff in fnames:
    # DEBUG
    print 'At time %03d of %03d processing %s ' % (Time + 1, len(fnames), ff)

    LL = []
    of = ApriFile(os.path.join(IDir, ff), 'rb')
    for ln in of:
        vvvv = ln.strip().split()
        LL.append(vvvv)
    of.close()
    for ll in LL:
        # Check that this is an acceptable edge and then update the neighbors list...
        if ll[0] != ll[1]:
            # Count one event...
            Events += 1.

            # Updating 'nshots' -> float # of tries before changing degree;
            Act_Com_Dic[ll[0]]['nshots'] += 1.

            # Check if it is a new neighbor and thus a new edge activated...
            if (DG[ll[0]][ll[1]] or DG[ll[1]][ll[0]]):
                if Act_Com_Dic[ll[0]]['company']:
                    # The activity class bin and the degree sub-bin...
                    act_class = Act_Com_Dic[ll[0]]['aa_bin']
                    deg_class = Act_Com_Dic[ll[0]]['kk_bin']
                    deg_alone_class = Act_Com_Dic[ll[0]]['degree_alone_bin']
                    degree = 0
                    if DG.__contains__(ll[0]):
                        degree = DG[ll[0]].deg()

                    # Updating the P_N_A dictionary for what concerns the
                    # events part that gets updated whenever a node makes
                    # a call...
                    P_N_A.setdefault(act_class, {})
                    P_N_A[act_class].setdefault(deg_class, {})
                    P_N_A[act_class][deg_class].setdefault(degree, {})
                    #P_N_A[act_class][deg_class][degree].setdefault('s_avg', .0)
                    #P_N_A[act_class][deg_class][degree].setdefault('s_wal', .0)
                    P_N_A[act_class][deg_class][degree].setdefault('s_new', .0)
                    P_N_A[act_class][deg_class][degree].setdefault('s_eve', .0)
                    # Here we update the events counter in the denominator.
                    P_N_A[act_class][deg_class][degree]['s_eve'] += 1.

                    # Updating the P_N_K dictionary for what concerns the
                    # events part that gets updated whenever a node makes
                    # a call...
                    P_N_K.setdefault(deg_alone_class, {})
                    P_N_K[deg_alone_class].setdefault(degree, {})
                    P_N_K[deg_alone_class][degree].setdefault('s_new', .0)
                    P_N_K[deg_alone_class][degree].setdefault('s_eve', .0)
                    # Here we update the events counter in the denominator.
                    P_N_K[deg_alone_class][degree]['s_eve'] += 1.
            else: # new edge case
                TOT_NUM_EDGES += 1.
                # Execute this part only if the caller is in-company...
                if Act_Com_Dic[ll[0]]['company']:
                    # The activity class bin and the degree sub-bin...
                    act_class = Act_Com_Dic[ll[0]]['aa_bin']
                    deg_class = Act_Com_Dic[ll[0]]['kk_bin']
                    deg_alone_class = Act_Com_Dic[ll[0]]['degree_alone_bin']

                    # We also save the degree and the previous shots for code clearness...
                    degree = 0
                    if DG.__contains__(ll[0]):
                        degree = DG[ll[0]].deg()
                    nshots = Act_Com_Dic[ll[0]]['nshots']

                    # Updating the P_N_A dictionary...
                    P_N_A.setdefault(act_class, {})
                    P_N_A[act_class].setdefault(deg_class, {})
                    P_N_A[act_class][deg_class].setdefault(degree, {})
                    #P_N_A[act_class][deg_class][degree].setdefault('s_avg', .0)
                    #P_N_A[act_class][deg_class][degree].setdefault('s_wal', .0)
                    P_N_A[act_class][deg_class][degree].setdefault('s_new', .0)
                    P_N_A[act_class][deg_class][degree].setdefault('s_eve', .0)

                    P_N_A[act_class][deg_class][degree]['s_new'] += 1.
                    P_N_A[act_class][deg_class][degree]['s_eve'] += 1.

                    #P_N_A[act_class][deg_class][degree]['s_avg'] += 1./nshots
                    #P_N_A[act_class][deg_class][degree]['s_wal'] += 1.

                    # Updating the P_N_K dictionary for what concerns the
                    # events part that gets updated whenever a node makes
                    # a call...
                    P_N_K.setdefault(deg_alone_class, {})
                    P_N_K[deg_alone_class].setdefault(degree, {})
                    P_N_K[deg_alone_class][degree].setdefault('s_new', .0)
                    P_N_K[deg_alone_class][degree].setdefault('s_eve', .0)
                    # Here we update both the new events counter and
                    # the denominator.
                    P_N_K[deg_alone_class][degree]['s_new'] += 1.
                    P_N_K[deg_alone_class][degree]['s_eve'] += 1.

                    ## Now append the event in the single story of the user...
                    #pna_singles.setdefault(act_class, {})
                    #pna_singles[act_class].setdefault(ll[0], {})
                    #pna_singles[act_class][ll[0]].setdefault('n', [])
                    #pna_singles[act_class][ll[0]].setdefault('pn', [])
                    #pna_singles[act_class][ll[0]].setdefault('dp', [])

                    #pna_singles[act_class][ll[0]]['n'].append(degree)
                    #pna_singles[act_class][ll[0]]['pn'].append(1./nshots)
                    #pna_singles[act_class][ll[0]]['dp'].append(1./np.sqrt(nshots))

                # Now check on the called...
                if Act_Com_Dic[ll[1]]['company'] and\
                        (Act_Com_Dic[ll[1]]['nshots'] > .0):
                    # The activity class bin and the degree sub-bin...
                    act_class = Act_Com_Dic[ll[1]]['aa_bin']
                    deg_class = Act_Com_Dic[ll[1]]['kk_bin']
                    degree = 0
                    if DG.__contains__(ll[1]):
                        degree = DG[ll[1]].deg()
                    nshots = Act_Com_Dic[ll[1]]['nshots']

                    # Updating the P_N_A dictionary...
                    P_N_A.setdefault(act_class, {})
                    P_N_A[act_class].setdefault(deg_class, {})
                    P_N_A[act_class][deg_class].setdefault(degree, {})
                    #P_N_A[act_class][deg_class][degree].setdefault('s_avg', .0)
                    #P_N_A[act_class][deg_class][degree].setdefault('s_wal', .0)
                    P_N_A[act_class][deg_class][degree].setdefault('s_new', .0)
                    P_N_A[act_class][deg_class][degree].setdefault('s_eve', .0)

                    #P_N_A[act_class][deg_class][degree]['s_wal'] += 1.

                    ## Now append the event in the single story of the user...
                    #pna_singles.setdefault(act_class, {})
                    #pna_singles[act_class].setdefault(ll[1], {})
                    #pna_singles[act_class][ll[1]].setdefault('n', [])
                    #pna_singles[act_class][ll[1]].setdefault('pn', [])
                    #pna_singles[act_class][ll[1]].setdefault('dp', [])

                    #pna_singles[act_class][ll[1]]['n'].append(degree)
                    #pna_singles[act_class][ll[1]]['pn'].append(.0)
                    #pna_singles[act_class][ll[1]]['pn'].append(1./np.sqrt(nshots))

                # We reset the event counter to 0 for both the involved nodes (the called
                # has now changed degree without being active...)
                Act_Com_Dic[ll[0]]['nshots'] = .0
                Act_Com_Dic[ll[1]]['nshots'] = .0

            # We now update the edge's weight regardless of the event kind...
            DG[ll[0]][ll[1]] += 1.

    # Increase the time in terms of analyzed files...
    Time += 1.

    # Now we compute the average degree (average over the active nodes so far)
    # and the P(a,k,t) if the time is within our selected list...
    active_nodes = len(DG)
    active_company_nodes = .0
    Net_Time.append(Time)

    # The average degree per activity class...
    for ktemp in DG:
        #for k1 in DG[ktemp]:
            #if DG[ktemp][k1] and DG[k1][ktemp]:
                #TOT_NUM_EDGES += .5
            #elif DG[ktemp][k1] and (not DG[k1][ktemp]):
                #TOT_NUM_EDGES += 1.

        if Act_Com_Dic[ktemp]['company']:
            active_company_nodes += 1.
            # The activity class bin...
            act_class = Act_Com_Dic[ktemp]['aa_bin']

            # Setting defaults and updating...
            k_a_t.setdefault(act_class, {})
            k_a_t[act_class].setdefault(Time, .0)
            k_a_t[act_class][Time] += float(DG[ktemp].deg())

            n_a_t.setdefault(act_class, {})
            n_a_t[act_class].setdefault(Time, 0.)
            n_a_t[act_class][Time] += 1.

            # Now update the P(a,k,t) if the time is suitable...
            if Time in TVec:
                # Setting defaults and updating...
                Pakt.setdefault(act_class, {})
                Pakt[act_class].setdefault(Time, {})
                Pakt[act_class][Time].setdefault('bb', [])
                Pakt[act_class][Time].setdefault('kk', [])
                Pakt[act_class][Time]['kk'].append(float(DG[ktemp].deg()))

    # We now compress the P(a,k,t) in real time...
    if Time in TVec:
        # DEBUG
        print '\t ... and the P(a,k,t) compression...'

        for PaK in Pakt.keys():
            if Time in Pakt[PaK].keys():
                if False: #Classic, unhandy version...
                    if sum(Pakt[PaK][Time]['kk']) > 0:
                        kmax = max(Pakt[PaK][Time]['kk'])*1.001
                        kmin = max(.999, .999*min(Pakt[PaK][Time]['kk']))
                        Temp_Bins = np.logspace( np.log10(kmin), np.log10(kmax),\
                                np.ceil(np.log(kmax/kmin)/np.log(1.1)) + 1)
                        Pakt[PaK][Time]['bb'] = Lin_Log_Bins(Temp_Bins)
                        Pakt[PaK][Time]['kk'], Pakt[PaK][Time]['bb'] =\
                                np.histogram(Pakt[PaK][Time]['kk'],\
                                bins=Pakt[PaK][Time]['bb'], density=True)
                    else:
                        Pakt[PaK][Time]['bb'] = [.0, 1.]
                        Pakt[PaK][Time]['kk'], Pakt[PaK][Time]['bb'] =\
                                np.histogram(Pakt[PaK][Time]['kk'],\
                                bins=Pakt[PaK][Time]['bb'])
                else:
                    ks = [int(k) for k in Pakt[PaK][Time]['kk']]
                    kmax = max(ks)
                    Pakt[PaK][Time]['bb'] = range(kmax+1)
                    Pakt[PaK][Time]['kk'] = [.0]*(kmax+1)
                    for k in ks:
                        Pakt[PaK][Time]['kk'][k] += 1.

    # The number of active edges...
    E_active_t.append(TOT_NUM_EDGES)

    # The number of active nodes...
    N_active_t.append(float(active_nodes))

    # And the number of events so far...
    N_events_t.append(Events)

    # The average degree 2E/N...
    K_events_t.append(TOT_NUM_EDGES*2./active_company_nodes)


#################################################################
### SAVING
#################################################################

## Saving the single stories...
#
#foutdat = gzip.open(ODir + 'data/p_n_single_stories_' + LABEL + '.dat', 'wb')
#cPickle.dump(N_A_K, foutdat)
#cPickle.dump(pna_singles, foutdat)
#foutdat.close()
# Freeing some space...
pna_singles = None

print 'Single stories saved in %s ...' % (ODir + 'data/p_n_single_stories_' + LABEL + '.dat')

# Now saving all the distributions and vectors...

# Temporary vectors to store the weights of the single edges...
Edges_Weight_Vec = []

# Containers for the activity based distribution of the:
# Weight;
Pw = {}
# Degree;
Pk = {}

# Big mama...
DATA = {}
DATA['TOT'] = {}
DATA['ACT'] = {}

DATA['TOT']['Stats'] = Network_Stats
DATA['TOT']['N_A_K'] = N_A_K
DATA['TOT']['N_K'] = N_K

DATA['TOT']['Vectors'] = {}
DATA['TOT']['Vectors']['aa'] = np.zeros(SOC_COMPANY_USERS)
DATA['TOT']['Vectors']['kk'] = np.zeros(SOC_COMPANY_USERS)
DATA['TOT']['Vectors']['kin'] = np.zeros(SOC_COMPANY_USERS)
DATA['TOT']['Vectors']['kout'] = np.zeros(SOC_COMPANY_USERS)
DATA['TOT']['Vectors']['win'] = np.zeros(SOC_COMPANY_USERS)
DATA['TOT']['Vectors']['wout'] = np.zeros(SOC_COMPANY_USERS)

# The cycle through all the nodes...
# ... and the counter of company users found so far...
ii = 0
for Nod in DG:
    if Act_Com_Dic[Nod]['company']:
        DATA['TOT']['Vectors']['kk'][ii] = DG[Nod].deg()
        DATA['TOT']['Vectors']['kin'][ii] = DG[Nod].inDeg()
        DATA['TOT']['Vectors']['kout'][ii] = DG[Nod].outDeg()

        temp_a = .0
        temp_win = .0
        temp_wout = .0
        for Nod1 in DG[Nod].iterIn():
            temp_win += DG[Nod1][Nod]
        for Nod1 in DG[Nod].iterOut():
            temp_wout += DG[Nod][Nod1]
            temp_a += DG[Nod][Nod1]

        DATA['TOT']['Vectors']['aa'][ii] = temp_a
        DATA['TOT']['Vectors']['win'][ii] = temp_win
        DATA['TOT']['Vectors']['wout'][ii] = temp_wout

        act_class = Act_Com_Dic[Nod]['aa_bin']
        Pw.setdefault(act_class, {})
        Pw[act_class].setdefault('in', {})
        Pw[act_class].setdefault('out', {})
        Pw[act_class]['in'].setdefault('b', [])
        Pw[act_class]['in'].setdefault('w', [])
        Pw[act_class]['out'].setdefault('b', [])
        Pw[act_class]['out'].setdefault('w', [])

        Pw[act_class]['in']['w'].append(temp_win)
        Pw[act_class]['out']['w'].append(temp_wout)


        Pk.setdefault(act_class, {})
        Pk[act_class].setdefault('tot', {})
        Pk[act_class].setdefault('in', {})
        Pk[act_class].setdefault('out', {})
        Pk[act_class]['tot'].setdefault('b', [])
        Pk[act_class]['tot'].setdefault('k', [])
        Pk[act_class]['in'].setdefault('b', [])
        Pk[act_class]['in'].setdefault('k', [])
        Pk[act_class]['out'].setdefault('b', [])
        Pk[act_class]['out'].setdefault('k', [])

        Pk[act_class]['tot']['k'].append(DG[Nod].deg())
        Pk[act_class]['in']['k'].append(DG[Nod].inDeg())
        Pk[act_class]['out']['k'].append(DG[Nod].outDeg())

        # Counting one more company user...
        ii += 1


# Now compress the Pw and Pk...
for ak in Pw.keys():
    for kind in Pw[ak].keys():
        if len(Pw[ak][kind]['w']) > 0:
            Pw[ak][kind]['b'] = np.arange(max(Pw[ak][kind]['w']) + 1.)
            Pw[ak][kind]['w'], Pw[ak][kind]['b'] =\
                    np.histogram(Pw[ak][kind]['w'], Pw[ak][kind]['b'], density=True)

for ak in Pk.keys():
    for kind in Pk[ak].keys():
        if len(Pk[ak][kind]['k']) > 0:
            Pk[ak][kind]['b'] = np.arange(max(Pk[ak][kind]['k']) + 1.)
            Pk[ak][kind]['k'], Pk[ak][kind]['b'] =\
                    np.histogram(Pk[ak][kind]['k'], Pk[ak][kind]['b'], density=True)

# The total average degree, active nodes and active edges per time (real and events)...

DATA['TOT']['tt'] = Net_Time
DATA['TOT']['te'] = N_events_t
DATA['TOT']['TVec'] = TVec
DATA['TOT']['kt'] = K_events_t
DATA['TOT']['et'] = E_active_t
DATA['TOT']['Nt'] = N_active_t

# Now the degree and activity distribution...
aa_s = copy.deepcopy(DATA['TOT']['Vectors']['aa'])
kk_s = copy.deepcopy(DATA['TOT']['Vectors']['kk'])

aa_s = np.array(aa_s)
kk_s = np.array(kk_s)

aa_s = aa_s[aa_s != .0]
kk_s = kk_s[kk_s != .0]

aa_s /= aa_s.max()

aa_bins = np.logspace(np.log10(aa_s.min()*.999), np.log10(aa_s.max()*1.001), 40)
kk_bins = np.logspace(np.log10(max(.999, kk_s.min()*.999)), np.log10(kk_s.max()*1.001), 40)
kk_bins = Lin_Log_Bins(kk_bins)

DATA['TOT']['rho_a'] = {}
DATA['TOT']['rho_a']['rho_a'], DATA['TOT']['rho_a']['bin'] = np.histogram(aa_s,\
        aa_bins, density=True)

DATA['TOT']['rho_k'] = {}
DATA['TOT']['rho_k']['rho_k'], DATA['TOT']['rho_k']['bin'] = np.histogram(kk_s,\
        kk_bins, density=True)

# Freeing some space...
aa_s = None
kk_s = None

DATA['ACT']['Bins'] = Bins
DATA['ACT']['k_a_t'] = k_a_t
DATA['ACT']['n_a_t'] = n_a_t
DATA['ACT']['P_akt'] = Pakt
DATA['ACT']['Nshots'] = Nshots
DATA['ACT']['Njumps'] = Njumps
DATA['ACT']['P_N_A'] = P_N_A
DATA['ACT']['P_N_K'] = P_N_K
DATA['ACT']['Pw'] = Pw
DATA['ACT']['Pk'] = Pk

foutdat = gzip.open(ODir + 'data/DATA_' + LABEL + '.dat', 'wb')
cPickle.dump(DATA, foutdat)
foutdat.close()

print 'Data saved in:%s' % (ODir + 'data/DATA_' + LABEL + '.dat')

