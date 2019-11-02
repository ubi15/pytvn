
def Analyze(DG, Time, PAKT, KAT):
    # The average out_degree per activity class...
    for nod in DG.values():
        if not (nod['A'] and nod['C']):
            continue
        # The activity class bin...
        act_class = nod['a']
        deg_class = nod['k']
        tmp_degree = nod['deg']

        # Setting defaults and updating...
        Pakt = PAKT[act_class][Time]

        if len(Pakt['k']) == 0:
            Pakt['k'].append(tmp_degree)
            Pakt['n'].append(0)
        else:
            while tmp_degree < Pakt['k'][0]:
                Pakt['k'].insert(0, Pakt['k'][0]-1)
                Pakt['n'].insert(0, 0)
            while tmp_degree > Pakt['k'][-1]:
                Pakt['k'].append(Pakt['k'][-1]+1)
                Pakt['n'].append(0)
        Pakt['n'][Pakt['k'].index(tmp_degree)] += 1

    for act_class, act_dict in KAT.items():
        if PAKT[act_class][Time]['n']:
            act_dict[Time] = sum([deg*num for deg, num in zip(PAKT[act_class][Time]['k'], PAKT[act_class][Time]['n'])])/\
                             float(sum([num for num in PAKT[act_class][Time]['n']]))


def Clustering_Coeff(G, perc=.2):
    '''
    Computes the unweighted clustering coefficient for the graph among the
    specified percent *perc* of nodes [perc=.25]
    '''
    from numpy.random import rand
    from numpy import ceil

    to_sample = [k for k, nod in G.items() if nod['A'] and nod['C'] if rand()<perc]

    summa = .0
    num_act = .0
    for nod_id in to_sample:
        nod = G[nod_id]
        if nod['C'] and nod['A']:
            summa += Clustering_node(G, nod_id)
            num_act += 1.

    return summa/max(1., num_act)

def Clustering_node(G, nod):
    deg = len(G[nod]['n'])
    if deg <= 1:
        return .0

    max_links = deg*(deg-1.)/2.

    vicini = set([k for k,v in G[nod]['n'].items() if v['A']])

    links = .0
    for neighb in vicini:
        for n_of_n, edge in G[neighb]['n'].items():
            if n_of_n in vicini and edge['A']:
                links += .5

    return links/max_links

