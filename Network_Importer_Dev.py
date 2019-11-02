import os
import sys
import gzip
import cPickle
import numpy as np

from my_foos import Lin_Log_Bins, nestedBins
from Analyze_Tools import AnalyzeNew, Clustering_Coeff, clusteringCoefficientTriangles

def Network_Importer(IDir, ODir, step_by_step=True, n_t_smpl=20, time_scheme="eve", binning_scheme="ak",\
                    act_bins_factor=1.2, deg_bins_factor=1.2, entr_bins_factor=2., first_entr_t=None,\
                    zipped_f=False, clust_perc=.1, starting_time=.001, file_ini=0, file_fin=None,\
                    caller_idx=0, called_idx=1, clr_company_idx=None, cld_company_idx=None, time_idx=None,\
                    max_loaded_events=100000000, saveNetworkDump=False, givenBins=None):
    '''
    Analyze and import a sequence of contacts.

    Usage:
    Network_Importer(IDir, ODir, SocTHr=0, step_by_step=True, n_t_smpl=10, TIME_events=True, **kwargs)

    Parameters
    ----------
    IDir: string
       path to the input directory;

    ODir: string
        path to the output directory;

    step_by_step: bool
        whether or not to use the step by step importing rather than the aggregated view per file. Default is True.

    n_t_smpl: int
        number of log-spaced times to analyze: (if no time is specified in the input files time will be measured
        according to time_events). Default is 10.

    time_scheme: ["eve"], "given"
        Define time as:
            - "eve": number of events (counting number of lines);
            - "given": use the column specified in `time_idx` as reference time;

   starting_time: [.05] float
        The first time at which we analyze the P(a,k,t) and the clustering coefficient. Default .05 if event time/time specified (i.e. we start after 5% of the total time has passed).

   binning: str
        ['aek']: the nested bins structure to compute, we can choose between {"e": 'entrance', "a": 'activity', "k": 'degree'}. If `givenBins` is passed and it
        is not `None` this variables is used to give the name to the output file and nothing more.

    givenBins: dict [None]
        If passed, it must be a dictionary whith two key-value copules:
            "labels":   dictionary whose keys contain ALL of the company nodes and whose
                        values are the already computed classes assigned to each node in
                        a tuple fashion, i.e., `(level0, level1, level2)`.
            "bins":     dictionary mimicking the dictionary returned by `nestedBins`, i.e., for
                        each level of the nodes labels it must contain the bins and the child
                        subclasses' details:

                        {
                            "b": [np.array] # the bins of the first level
                            "v": { # Contains information on the second-level subclasses
                                    i: { # i-th subclass faling in the i-th bin of the upper class
                                        "b": [np.array] # bins of the second level for this subclass
                                        "v": {i: {third level} for i in second level bins}
                                    }
                                }
                        }
        When passed it forces the program to skip the binning procedure.

   act_bins_factor: float
        The factor of the activity log-bins. Default is 1.25 .

   deg_bins_factor: float
        The factor of the degree log-bins. Default is 1.25 .

   entr_min: int
        The first bin of the entrance time goes from 1 to entr_min. Default is 10000.

   entr_bins_factor: float
        The factor for the entrance time binning. Default is 2.

   tini: int
        The analysis will begin from the tini-th file in the folder. Default is `0`.

   tfin: int
        The analysis will end at the tfin-th file in the folder. Default is `Number_of_files`.

   zipped_f: bool
        Whether or not the files in the folder are zipped. Default is `True`.

   clust_perc: float
        The percentual of nodes on which to compute the clustering coefficient. Default is 0.2 (i.e. 20%).

    The file expects an input file structured as follows:
    CALLER_ID \t CALLED_ID \t COMPANY_CLR \t COMPANY_CLD \t TIME_event \n

    The last column is optional and sets the time of reference of the event. If not given we will
    assume the following time:
        - each event is at time t_i=number_of_lines if the number of files < 2*n_t_smpl;
        - each event inside a single file is at time #num_of_file;
    '''

    # Input and output directories...
    assert(isinstance(IDir, str) and os.path.exists(IDir) and os.path.isdir(IDir))
    assert(isinstance(ODir, str))
    assert(isinstance(n_t_smpl, int) and n_t_smpl > 0) # The number of time values to analyze...
    assert(isinstance(starting_time, float) and (.001 < starting_time < .95) )

    ApriFile = gzip.open if zipped_f else open # Gzipped input?

    # The list of file names to process...
    fnames = sorted([f for f in sorted(os.listdir(IDir)) if f[0] != "."])

    # The initial and final time (default 0 -> number of files to be analyzed...)...
    tini = file_ini
    tfin = len(fnames) if not file_fin else file_fin
    # Then we select only the right slice of files to be analyzed...
    fnames = fnames[tini:tfin]

    # Creating ODir
    tmPath = ""
    for d in os.path.split(ODir):
        tmPath = os.path.join(tmPath, d)
        if not os.path.exists(tmPath) or not os.path.isdir(tmPath):
            os.mkdir(tmPath)
    for additional_path in ['', 'data']:
        ODir = os.path.join(ODir, additional_path)
        if (not os.path.exists(ODir)) or (not os.path.isdir(ODir)):
            os.mkdir(ODir)

    assert(isinstance(clust_perc, float) and (.0 <= clust_perc <= 1.))
    assert(isinstance(act_bins_factor, float) and (act_bins_factor> 1.))
    assert(isinstance(deg_bins_factor, float) and (deg_bins_factor> 1.))

    # Line fetcher
    fetcherClrCld = lambda vs: (vs[caller_idx], vs[called_idx],\
                                True, True, -1)
    fetcherClrCom = lambda vs: (vs[caller_idx], vs[called_idx],\
                                bool(int(vs[clr_company_idx])), bool(int(vs[cld_company_idx])), -1)
    fetcherClrTim = lambda vs: (vs[caller_idx], vs[called_idx],\
                                True, True, float(vs[time_idx]))
    fetcherClCoTi = lambda vs: (vs[caller_idx], vs[called_idx],\
                                bool(int(vs[clr_company_idx])), bool(int(vs[cld_company_idx])), float(vs[time_idx]))

    if clr_company_idx and cld_company_idx:
        if time_idx:
            print "Using time and company fetcher"
            lineFetcher = fetcherClCoTi
        else:
            print "Using company fetcher"
            lineFetcher = fetcherClrCom
    else:
        if time_idx:
            print "Using time fetcher"
            lineFetcher = fetcherClrTim
        else:
            print "Using clr and cld fetcher"
            lineFetcher = fetcherClrCld

    # The reference times
    time_ini, time_fin = 0, 0

    # Self explaining (the files stores in each row: "CallerID CalledID companyclr companycld time\n")...
    with ApriFile(os.path.join(IDir, fnames[0]), 'rb') as ref_file:
        print 'Reference file: ', fnames[0]
        print 'Last file: ', fnames[-1]
        reference_values = ref_file.readline().strip().split()
        num_vals = len(reference_values)
        print "read the first line: ", reference_values
        print "Corresponding to:"
        print "CLR", reference_values[caller_idx],
        print "CLD", reference_values[called_idx],
        if clr_company_idx:
            print "CLR company", reference_values[clr_company_idx],
        if cld_company_idx:
            print "CLD company", reference_values[cld_company_idx],
        if time_idx:
            print "Time", float(reference_values[time_idx])
            if time_scheme == "given":
                time_ini = reference_values[time_idx]

        print "Mapped to ", lineFetcher(reference_values)

    # Define the per-edge and per-node features...
    featPerNode, featPerEdge = 8, 4
    nActive, nCompany, nClass, nDegree, nAct, nTact, nNeighbors, nShots = range(featPerNode)
    eActive, eStrength, eNTriangles, eTact = range(featPerEdge)

    # For over the files...
    EV_TOT = 0
    MyGraph = {}
    for file_index, fn in enumerate(fnames):
        sys.stdout.write("\rFirst round, file %03d of %03d..." % (file_index+1, len(fnames)) )
        sys.stdout.flush()

        sliceNum = 0
        readDone = False
        while not readDone:
            Listone = []
            with ApriFile(os.path.join(IDir, fn), 'rb') as of:
                for _ in xrange(max_loaded_events*sliceNum): next(of)

                for ln in of:
                    lineValues = ln.strip().split()
                    Listone.append(lineFetcher(lineValues))

                    if len(Listone) >= max_loaded_events:
                        sliceNum += 1
                        break
                else:
                    readDone = True
            for clr, cld, comp_clr, comp_cld, time in Listone:
                if clr != cld:
                    EV_TOT += 1
                    time_fin = time if time_scheme == "given" else EV_TOT
                    if clr not in MyGraph:
                        MyGraph[clr] = [False, comp_clr, (), 1, 1, EV_TOT, {cld: [False, 1, 0, EV_TOT]}, .0]
                    else:
                        try:
                            MyGraph[clr][nNeighbors][cld][eStrength] += 1
                        except KeyError:
                            MyGraph[clr][nNeighbors][cld] = [False, 1, 0, EV_TOT]
                            MyGraph[clr][nDegree] += 1
                        MyGraph[clr][nAct] += 1

                    if cld not in MyGraph:
                        MyGraph[cld] = [False, comp_cld, (), 1, 0, EV_TOT, {clr: [False, 0, False, 0, EV_TOT]}, .0]
                    else:
                        if clr not in MyGraph[cld][nNeighbors]:
                            MyGraph[cld][nNeighbors][clr] = [False, 0, False, 0, EV_TOT]
                            MyGraph[cld][nDegree] += 1
            del Listone

    TOT_COMPANY_USERS, TOT_NUM_EDGES = 0, .0
    for nod in MyGraph.itervalues():
        companyFrom = nod[nCompany]
        for neighb in nod[nNeighbors].iterkeys():
            if companyFrom or MyGraph[neighb][nCompany]:
                TOT_NUM_EDGES += .5
        if companyFrom:
            TOT_COMPANY_USERS += 1
    print ''
    print 'Total Network:'
    print ''
    print 'Total Nodes: %1.03e' % len(MyGraph)
    print 'Company Nodes: %1.03e' % TOT_COMPANY_USERS
    print 'Edges (at least one vertex is in-community): %1.03e' % TOT_NUM_EDGES
    print 'Events: %1.03e' % EV_TOT
    sys.stdout.write("Total size of graph: %d Bytes\n" % sys.getsizeof(MyGraph))
    print ''

    # We then initialize the LABEL that will be given to all the output files...
    period = '%03d-%03d' % (tini, tfin)
    LABEL = '_a%03d_k%03d_e%03d_stBySt%s_tSch%s_bSch%s' %\
            (act_bins_factor*100, deg_bins_factor*100, entr_bins_factor*100, int(step_by_step), time_scheme, binning_scheme)

    ## We now initialize the vectors to define the bins...
    IDs = ["" for i in range(TOT_COMPANY_USERS)]
    Activities, Degrees, Entrances, Ks_in, Ks_out, Ws_in = np.zeros(TOT_COMPANY_USERS), np.zeros(TOT_COMPANY_USERS),\
        np.zeros(TOT_COMPANY_USERS), np.zeros(TOT_COMPANY_USERS), np.zeros(TOT_COMPANY_USERS), np.zeros(TOT_COMPANY_USERS)
    iiiNodeCount = 0
    for nodeID, nodeVals in MyGraph.iteritems():
        if not nodeVals[nCompany]:
            continue
        IDs[iiiNodeCount] = nodeID
        Activities[iiiNodeCount] = nodeVals[nAct]
        Degrees[iiiNodeCount] = nodeVals[nDegree]
        Entrances[iiiNodeCount] = nodeVals[nTact]

        # k_in, k_out and w_in (w_out is the activity)...
        k_in, k_out, w_in = 0, 0, 0
        for neighb, edge in nodeVals[nNeighbors].iteritems():
            inStr = MyGraph[neighb][nNeighbors][nodeID][eStrength]
            k_in += min(1, inStr)
            w_in += inStr
            k_out += min(1, edge[eStrength])
        Ks_in[iiiNodeCount] = k_in
        Ks_out[iiiNodeCount] = k_out
        Ws_in[iiiNodeCount] = w_in
        iiiNodeCount += 1
    # The true activity based only on the events involving only actual social nodes...
    print "EVTOT, EVTOT_sumA", EV_TOT, sum(Activities)

    if not givenBins:
        # Binning the nodes...
        Bins = nestedBins(binning_scheme, Activities, Degrees, Entrances,\
                            act_bins_factor, deg_bins_factor, entr_bins_factor, first_entr_t)
        sys.stdout.write("Binning nodes...")
        i = 0
        for nodeID, nodeVals in MyGraph.iteritems():
            if not nodeVals[nCompany]:
                continue
            tmp_class = []
            b0, b1, b2 = 0, 0, 0
            for level, what in enumerate(binning_scheme):
                tmp_val = nodeVals[nAct] if what == "a" else nodeVals[nDegree] if what == "k" else nodeVals[nTact]
                if level == 0:
                    b0 = np.argmax(Bins["b"] > tmp_val)-1
                elif level == 1:
                    b1 = np.argmax(Bins["v"][b0]["b"] > tmp_val)-1
                elif level == 2:
                    b2 = np.argmax(Bins["v"][b0]["v"][b1]["b"] > tmp_val)-1
            nodeVals[nClass] = (b0, b1, b2) if len(binning_scheme) == 3 else (b0, b1) if len(binning_scheme) == 2 else (b0)
            i += 1
            if i % 100000 == 0:
                sys.stdout.write("\r%09d/%09d..." % (i, len(MyGraph)))
                sys.stdout.flush()
        sys.stdout.write("\r%09d/%09d..." % (i, len(MyGraph)))
        sys.stdout.flush()
        sys.stdout.write("\nDone!\n")
    else:
        # Initialize the nodes with the classes passed in `givenBins`.
        from copy import deepcopy
        Bins = deepcopy(givenBins["bins"])
        givenLabels = deepcopy(givenBins["labels"])

        for nodeIndex, nodeVals in MyGraph.iteritems():
            if not nodeVals[nCompany]:
                continue
            try:
                nodeVals[nClass] = givenLabels[nodeIndex]
            except KeyError:
                sys.stdout.write("ERROR when initializing the nodes classes, node %r not in `givenBins['labels']` keys.\n" % (nodeIndex))
                return 2

    ################################################################
    # Dictionary that stores all the p_n_related stuff in one
    # structure only, the structure is as follows:
    # P_N_A[class][k] = {'s_new': .0, 's_eve': .0}
    # where:
    # - 's_new' is the counter of events toward new nodes by nodes at degree n;
    # - 's_eve' is the counter of the total events done by any node at degree n;
    # so that the p_n_events will be defined as: `p_n_ev(n, ak, dk) = s_new/s_eve`
    # Generate the times to analyze for the P(a,k,t)...
    TVec = np.logspace(np.log10(time_ini + (time_fin - time_ini)*starting_time), np.log10(time_fin), n_t_smpl)
    TVec = np.unique(TVec)
    TVec.sort()
    # And check that the last one is correct...
    TVec[-1] = float(time_fin)
    print "Times to analyze = ", TVec

    # Classes to save:
    nodesClasses = set([nodeVals[nClass] for nodeID, nodeVals in MyGraph.iteritems() if nodeVals[nCompany]])
    P_N_A = {nodeClass: {} for nodeClass in nodesClasses}
    # The dictionary that contains the P(a, k, t)
    Pakt = {nodeClass: {t: {} for t in range(len(TVec))} for nodeClass in nodesClasses}
    # We also count the population in each bin and sub bin...
    N_A_K = {nodeClass: 0 for nodeClass in nodesClasses}

    # Time list...
    # And the number of events so far...
    Events_t = []
    # Real time used here (practically a replica of TVec)
    Real_Time_t = []
    # The number of active nodes per time...
    Nodes_active_t = []
    # The number of active edges per time...
    Edges_active_t = []
    # Plus the average degree up to time t for the whole network = 2E/N
    K_mean_t = []
    # The clustering coefficient(t)
    Clust_t = []
    # Events that:
    newCloseTriang = np.array([0 for t in TVec])   # New edge closes one or more triangles
    newOpenTriang =  np.array([0 for t in TVec])   # New edge closes no triangles
    oldCloseTriang = np.array([0 for t in TVec])   # Old edge that insist on one or more closed triangles
    oldOpenTriang =  np.array([0 for t in TVec])   # Old edge that insists on no closed triangles...

    # Last cycle to count the nodes per class and reset all the variables for the network evolution
    # (like we reset the degee to 0)
    for nodeID, nodeVals in MyGraph.iteritems():
        if nodeVals[nCompany]:
            N_A_K[nodeVals[nClass]] += 1
        nodeVals[nDegree] = 0

    # Save the network Statistic...
    networkStats = {'Stats': {'Nodes': len(MyGraph), 'CompanyNodes': TOT_COMPANY_USERS,\
                            'Edges': TOT_NUM_EDGES, 'Events': int(EV_TOT)},\
                    'Arrays': {"Act": Activities, "Deg": Degrees, "Entr": Entrances,\
                            "Kin": Ks_in, "Kout": Ks_out, "Win": Ws_in, "IDs": IDs},\
                    'Bins': {"Bins": Bins, "N_A_K": N_A_K}
                    }
    # Time and events counters...
    next_time = 0
    Events = 0
    TOTactiveEdges = 0 # The growing number of edges...
    for file_index, fn in enumerate(fnames):
        sys.stdout.write("\rSecond round, file %03d of %03d..." % (file_index+1, len(fnames)) )
        sys.stdout.flush()

        sliceNum = 0
        readDone = False
        while not readDone:
            Listone = []
            with ApriFile(os.path.join(IDir, fn), 'rb') as of:
                for _ in xrange(max_loaded_events*sliceNum): next(of)

                for ln in of:
                    lineValues = ln.strip().split()
                    Listone.append(lineFetcher(lineValues))

                    if len(Listone) >= max_loaded_events:
                        sliceNum += 1
                        break
                else:
                    readDone = True

            if step_by_step: # one event at once version...
                for clr, cld, comp_clr, comp_cld, time in Listone:
                    if clr == cld:
                        continue
                    Events += 1

                    nodeVals = MyGraph[clr]
                    edgeNeighborVals = nodeVals[nNeighbors][cld]
                    if nodeVals[nCompany]:
                        # The activity and degree bins and the current k...
                        degree =     nodeVals[nDegree]
                        tmp_pna = P_N_A[nodeVals[nClass]]
                        tmp_pna.setdefault(degree, {'s_new': .0, 's_eve': .0})

                    if edgeNeighborVals[eActive]:
                        if nodeVals[nCompany]:
                            tmp_pna[degree]['s_eve'] += 1.
                        # Old edge check triangles...
                        if edgeNeighborVals[eNTriangles] > 0:
                            oldCloseTriang[next_time] += 1
                        else:
                            oldOpenTriang[next_time] += 1
                    else:
                        if nodeVals[nCompany]:
                            tmp_pna[degree]['s_new'] += 1.
                            tmp_pna[degree]['s_eve'] += 1.
                        cldVals = MyGraph[cld]
                        # Execute this part whatever the caller company...
                        # We set both the direction of the edge as active...
                        edgeNeighborVals[eActive] = True
                        cldVals[nNeighbors][clr][eActive] = True
                        # We set the node as active and we increment the degree by 1...
                        nodeVals[nActive] = True
                        nodeVals[nDegree] += 1
                        # The same for the called, since he has now changed degree without being active...
                        cldVals[nActive] = True
                        cldVals[nDegree] += 1
                        TOTactiveEdges += 1.

                        # Check on the triangles...
                        # Cycle through the active neighbors of cld actively linked to clr and check for active edges
                        commonNeighbs = set([neighb for neighb, edge in nodeVals[nNeighbors].iteritems() if edge[eActive]]).intersection(\
                                                set([neighb for neighb, edge in cldVals[nNeighbors].iteritems() if edge[eActive]]))
                        if len(commonNeighbs) > 0:
                            for commonNeighb in commonNeighbs:
                                # This newly created edge closed all of these triangles so we have to increment the counter
                                # here!
                                nodeVals[nNeighbors][commonNeighb][eNTriangles] += 1
                                cldVals[nNeighbors][commonNeighb][eNTriangles] += 1
                            # Since it is a new edge this is the initial number of triangles...
                            edgeNeighborVals[eNTriangles] = len(commonNeighbs)
                            cldVals[nNeighbors][clr][eNTriangles] = len(commonNeighbs)

                            newCloseTriang[next_time] += 1
                        else:
                            newOpenTriang[next_time] += 1

                    # Here we check only if we got the time or we are counting by events...
                    do_analysis = False
                    if time_scheme == "given":
                        if time > TVec[next_time] or Events == EV_TOT:
                            do_analysis = True
                    elif time_scheme == "eve" and Events >= TVec[next_time]:
                        do_analysis = True

                    if do_analysis:
                        print "\t ... and the P(a,k,t) analysis...",
                        AnalyzeNew(MyGraph, next_time, Pakt, nActive, nCompany, nDegree, nClass)
                        # time in events so far...
                        Events_t.append(Events)
                        # Real time used here (practically a replica of TVec)
                        Real_Time_t.append(time)
                        # The number of active edges...
                        Edges_active_t.append(TOTactiveEdges)
                        # The number of active nodes...
                        nActiveCompanyNodes = sum([1 for nod in MyGraph.itervalues() if nod[nActive] and nod[nCompany]])
                        Nodes_active_t.append(nActiveCompanyNodes)
                        # The average degree 2E/N...
                        K_mean_t.append( TOTactiveEdges*2./float(nActiveCompanyNodes))
                        print " and the clustering coefficient..."
                        Clust_t.append(clusteringCoefficientTriangles(MyGraph, nActive, nCompany, nDegree, nNeighbors,\
                                                            eActive, eNTriangles, perc=clust_perc))

                        next_time += 1
                        if next_time >= len(TVec):
                            break
            else:
                #Integrated version
                DG_temp = {}    # For the integrated version...
                called = set()  # For the integrated version...
                for clr, cld, _, __, time in Listone:
                    if clr == cld:
                        continue
                    Events += 1

                    # Add cld to clr's neighbors...
                    called.add(cld)
                    DG_temp.setdefault(clr, {})
                    DG_temp[clr].setdefault('Out', {})
                    DG_temp[clr]['Out'].setdefault(cld, .0)
                    DG_temp[clr]['Out'][cld] += 1.
                    DG_temp[clr].setdefault('nshot', .0)
                    DG_temp[clr]['nshot'] += 1.

                # Now we delete from the called the non-company users and later the called with
                # no events because they could update their status in the first cycle...
                called = set([cld for cld in called if MyGraph[cld][nCompany]])

                # First cycle through the active nodes in this cycle to evaluate the p(n),
                # thus we cycle only through the active and company users...
                for nodeID, tmpNodeVals in DG_temp.iteritems():
                    nodeVals = MyGraph[nodeID]
                    if nodeVals[nCompany]:
                        # Number of new contacted nodes, number of
                        # events toward new neighbors and
                        # number of total calls...
                        New_n = False
                        Num_New_Eve = .0
                        Num_Tot_Eve = .0

                        # First check to count the fraction of new events and the total
                        # number of events...
                        for key_out, weight_out in tmpNodeVals['Out'].iteritems():
                            if not nodeVals[nNeighbors][key_out][eActive]:
                                New_n = True
                                Num_New_Eve += weight_out

                                # New edge check for triangles...
                                commonNeighbs = set([neighb for neighb, edge in nodeVals[nNeighbors].iteritems() if\
                                                    edge[eActive]]).intersection(\
                                        set([neighb for neighb, edge in MyGraph[key_out][nNeighbors].iteritems() if\
                                            edge[eActive]]))
                                if len(commonNeighbs) > 0:
                                    newCloseTriang[next_time] += 1
                                else:
                                    newOpenTriang[next_time] += 1
                            else:
                                # Old edge, check for triangles...
                                if nodeVals[nNeighbors][key_out][eNTriangles] > 0:
                                    oldCloseTriang[next_time] += 1
                                else:
                                    oldOpenTriang[next_time] += 1
                            Num_Tot_Eve += weight_out

                        # Updating the network's node with the total number of shots...
                        nodeVals[nShots] += Num_Tot_Eve

                        # The activity, degree classes bin and degree of node...
                        nodeClass, nodeDegree = nodeVals[nClass], nodeVals[nDegree]

                        # Updating the P_N_A dictionary for what concerns the events part
                        # that gets updated whenever a node makes a call...
                        P_N_A[nodeClass].setdefault(nodeDegree, {'s_new': .0, 's_eve': .0})
                        P_N_A[nodeClass][nodeDegree]['s_new'] += Num_New_Eve
                        P_N_A[nodeClass][nodeDegree]['s_eve'] += Num_Tot_Eve
                        if Num_Tot_Eve != tmpNodeVals['nshot']:
                            print "Houston we got a problem, nshots != Num_Tot_Eve in the"
                            exit(1)

                        # Check if I have new neighbors and remove clr from called because it
                        # changed degree on its own...
                        if New_n and (nodeID in called):
                            called.remove(nodeID)
                # We now remove from called the nodes that have no running calling events...
                called = set([cld for cld in called if MyGraph[cld][nShots] != .0])

                # Second cycle to update the total network...
                for nodeID, tmpNodeVals in DG_temp.iteritems():
                    for tmpNeighbID, tmpNeighbVals in tmpNodeVals['Out'].iteritems():
                        nodeVals = MyGraph[nodeID]
                        if not nodeVals[nNeighbors][tmpNeighbID][eActive]:
                            neighbNode = MyGraph[tmpNeighbID]
                            # Now we check if the neighbor neighb is in called:
                            # in that case we have to update its p_n_average.
                            if tmpNeighbID in called:
                                # The activity and degree classes bin and the degree...
                                neighClass, neighDegree, neighNShots =\
                                    neighbNode[nClass], neighbNode[nDegree], neighbNode[nShots]

                                # Updating the P_N_A dictionary for the called...
                                P_N_A[neighClass].setdefault(neighDegree, {'s_new': .0, 's_eve': .0})
                                P_N_A[neighClass][neighDegree]['s_eve'] += neighNShots

                                called.remove(tmpNeighbID)

                            # Check for triangles closure...
                            commonNeighbs = set([\
                                    neighb for neighb, edge in nodeVals[nNeighbors].iteritems() if edge[eActive]]).intersection(\
                                    set([neighb for neighb, edge in neighbNode[nNeighbors].iteritems() if edge[eActive]]))
                            if len(commonNeighbs) > 0:
                                for commonNeighb in commonNeighbs:
                                    nodeVals[nNeighbors][commonNeighb][eNTriangles] += 1
                                    neighbNode[nNeighbors][commonNeighb][eNTriangles] += 1
                                nodeVals[nNeighbors][tmpNeighbID][eNTriangles] = len(commonNeighbs)
                                neighbNode[nNeighbors][nodeID][eNTriangles] = len(commonNeighbs)

                            nodeVals[nNeighbors][tmpNeighbID][eActive] = True
                            nodeVals[nDegree] += 1
                            nodeVals[nShots] = .0
                            nodeVals[nActive] = True

                            neighbNode[nNeighbors][nodeID][eActive] = True
                            neighbNode[nDegree] += 1
                            neighbNode[nShots] = .0
                            neighbNode[nActive] = True

                            TOTactiveEdges += 1.

                # Here we check only if we got the time or we are counting by events...
                do_analysis = False
                if time_scheme == "given":
                    if time > TVec[next_time] or Events == EV_TOT:
                        do_analysis = True
                elif time_scheme == "eve" and Events >= TVec[next_time]:
                    do_analysis = True

                if do_analysis:
                    print "\t ... and the P(a,k,t) analysis...",
                    AnalyzeNew(MyGraph, next_time, Pakt, nActive, nCompany, nDegree, nClass)
                    # time in events so far...
                    Events_t.append(Events)
                    # Real time used here (practically a replica of TVec)
                    Real_Time_t.append(time)
                    # The number of active edges...
                    Edges_active_t.append(TOTactiveEdges)
                    # The number of active nodes...
                    nActiveCompanyNodes = sum([1 for nod in MyGraph.itervalues() if nod[nActive] and nod[nCompany]])
                    Nodes_active_t.append(nActiveCompanyNodes)
                    # The average degree 2E/N...
                    K_mean_t.append(TOTactiveEdges*2./float(nActiveCompanyNodes))
                    print " and the clustering coefficient..."
                    Clust_t.append(clusteringCoefficientTriangles(MyGraph, nActive, nCompany, nDegree, nNeighbors,\
                                                        eActive, eNTriangles, perc=clust_perc))
                    next_time += 1
                    if next_time >= len(TVec):
                        break
    ##########
    # SAVING #
    ##########
    sys.stdout.write("Done second cycle, now saving:\n")
    sys.stdout.write("\t- network stats...")
    sys.stdout.flush()
    networkStats["Params"] = {"act_bins_factor": act_bins_factor, "deg_bins_factor": deg_bins_factor,\
                            "entr_bins_factor": entr_bins_factor, "timeSampled": n_t_smpl, "binningScheme": binning_scheme,\
                            "clustPerc": clust_perc, "starting_time": starting_time, "first_entr_t": first_entr_t}
    networkStats["pna"] = P_N_A
    networkStats["pkt"] = Pakt
    networkStats["TimeVecs"] = {"TVec": TVec, "EventsT": Events_t, "RealTimeT": Real_Time_t,\
                    "NodesActiveT": Nodes_active_t, "EdgesActiveT": Edges_active_t, "KmeanT": K_mean_t,\
                    "Clust_t": Clust_t, "newCloseTriang": newCloseTriang, "newOpenTriang": newOpenTriang,\
                    "oldCloseTriang": oldCloseTriang, "oldOpenTriang": oldOpenTriang}
    outStatsFile = os.path.join(ODir, "networkStats%s.dat.gz" % LABEL)
    cPickle.dump(networkStats, gzip.open(outStatsFile, "wb"))
    sys.stdout.write(" done!\n")

    sys.stdout.write("\t- network dump...")
    sys.stdout.flush()
    if saveNetworkDump:
        cPickle.dump(MyGraph, gzip.open(os.path.join(ODir, "networkDump%s.dat.gz" % LABEL), "wb"))
    sys.stdout.write(" done!\n")

    return outStatsFile

