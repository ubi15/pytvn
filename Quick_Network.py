import numpy as np
import string
import cPickle
import copy
import os
import sys

def act_gen(N, epsilon, Amax, nu):
    return ( epsilon**(-nu) - np.random.rand(N) * ((epsilon**(-nu)) - (Amax**(-nu))) )**(-1./nu)


def Network_dyn(N=1000, epsilon=1e-3, nu=1.1, beta=.5, const=[(1., 1.)], steps=100000, stepsPerFile=10000, fracCompany=1.):
    G = NetX.DiGraph()
    # Some variables...
    use_net_a = True
    distbec = False
    distc = False
    distb = True
    aa = []
    junk = []
    if use_net_a:
        fin = open('../out/Sandbox_TWT_MyD/Social_Network_p-000-090_thres_00_a10_k01.dat')
        #fin = open('../out/Sandbox_YHO_EvH/Social_Network_p-000-028_thres_00_a10_k01.dat')
        junk = cPickle.load(fin)
        junk = cPickle.load(fin)
        junk = cPickle.load(fin)
        fin.close()
        acts = [junk.node[k]['aa'] for k in junk.nodes()]
        for i in range(int(N)):
            sel = np.random.randint(len(acts))
            aa.append(acts[sel])
            del acts[sel]
        MAXA = max(aa)
        aa = [a/MAXA for a in aa]
    else:
        aa = act_gen(N, ee, nu)
    for i, a in enumerate(aa):
        G.add_node(i, act=a)

    #betas = np.random.normal(bb, sb, N)
    betas = np.zeros(N, float)
    #betas = (aa/1e-1)**-.2
    const = np.zeros(N, float)
    placed = 0
    if distbec: # Both beta and c distributed...
        while betas[-1] == .0:
            x = np.random.lognormal(bb, sb, 1e+6)
            x = x[x <= hb]
            x = x[x >= lb]
            c = (x - lb)*(hc - lc)/(hb - lb) + lc
            for ind, i in enumerate(x):
                if placed == betas.size:
                    break
                betas[placed] = lb + hb - i
                const[placed] = c[ind]
                placed += 1
    elif distc: # only c distributed..
        while betas[-1] == .0:
            # Lognorm...
            x = np.random.lognormal(cc, sc, 1e+6)
            # Power...
            #x = ((hc**(1. - sc) - lc**(1. - sc))*np.random.uniform(.0, 1., 1e+6) + lc**(1. - sc))**(1./(1. - sc))
            x = x[x <= hc]
            x = x[x >= lc]
            for ind, i in enumerate(x):
                if placed == betas.size:
                    break
                betas[placed] = bb
                #betas[placed] = (aa[placed]/.1)**(-.5)
                const[placed] = i
                placed += 1
    elif distb: # only b distributed..
        while betas[-1] == .0:
            x = np.random.lognormal(bb, sb, 1e+6)
            x = x[x <= hb]
            x = x[x >= lb]
            for ind, i in enumerate(x):
                if placed == betas.size:
                    break
                betas[placed] = lb + hb - i
                const[placed] = cc
                placed += 1

    if distb or distbec:
        c, b, i = plt.hist(betas, 100, normed=True, align='mid')
        b = b[c!=.0]
        c = c[c!=.0]
        plt.yscale('log')
        plt.show()

    if distc or distbec:
        c, b, i = plt.hist(const, 100, normed=True, align='mid')
        b = b[c!=.0]
        c = c[c!=.0]
        plt.yscale('log')
        plt.show()


    #perc = .01
    #for i in range(int(np.ceil(N*perc))):
        #betas[randint(0, N - 1)] /= 10.

    for t in range(int(T)):
        fout = open('../data/nets_logn_short/nets_%04d.dat' % t, 'w')
        for i in range(int(N)):
            # Random selection each time...
            #att = int(floor(uniform(0., N)))
            # As in the for...
            att = i
            if uniform(0., 1.) <= G.node[att]['act']:
                nneigh = float(len(G.neighbors(att)))
                if (uniform(0., 1.) <= (const[att]/(const[att] + nneigh))**betas[att]):
                    sel = randint( 0., N - 1)
                    while sel in G.neighbors(i):
                        sel = randint(0., N - 1)
                    G.add_edge(att, sel)
                    print >> fout, att, sel
                else:
                    sel = randint(0., nneigh - 1)
                    print >> fout, att, G.neighbors(att)[sel]
        fout.close()
if __name__ == "__main__":
    N = 1e+5
    ee = 1e-2
    nu = 2.1
    bb = .1
    sb = 10.
    lb = .1
    hb = 10.
    cc = 2.
    sc = 2.
    lc = 1.
    hc = 50.
    T = 1e+2

    Network_dyn(N, ee, nu, bb, sb, lb, hb, cc, sc, lc, hc, T)

    print 'Done!'

    exit(0)


