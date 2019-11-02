import numpy as np
import sys


########################################
# Fitting function...
########################################

def my_p_n_scale(params, x, y, w):
    # Extracting the parameters...
    bbb = params['zz'].value
    scarti = []
    for index, par in enumerate('abcdefghijklmnopqr'):
        if params.has_key(par):
            ccc = params[par].value
            xxx = x[index, w[index, :].nonzero()]
            yyy = y[index, w[index, :].nonzero()]
            www = w[index, w[index, :].nonzero()]
            xxx.squeeze(0)
            yyy.squeeze(0)
            www.squeeze(0)
            model = (1. + xxx/ccc)**(-bbb)
            delta = (model - yyy)*www/yyy
            scarti.append(delta.sum())
    output = np.array(scarti)
    return output


def p_n_pow(pp, *args): # x, y = None, w = None):
    '''
    Fitting function $p(n) = (1+n/c)^{-\beta}$
    Usage:
    p_n_pow(pp=[beta, const], x, [values], [weights])

    If no values are provided the model estimation is returned. If no weights (STD deviation on the
    measured point) are given it return the normalized chi square, otherwise the weighted one.
    '''
    x = args[0]
    y = None
    w = None
    if len(args) == 2:
        y = args[1]
    if len(args) == 3:
        y = args[1]
        w = args[2]

    bb = pp[0]
    model =  (1. + x)**(-bb)
    if y is None:
        return model
    elif w is None:
        return (((model - y)/model)**2.).sum()
    else:
        return (((model - y)/w)**2.).sum()

def p_n_pow_const(pp, *args): # x, y = None, w = None):
    '''
    p_n_pow_const(pp = [beta, const], *args)

    Positional args are:
    x: the abscesses (mandatory);
    [y]: the ordinates (optional);
    [w]: the std-deviations of the ordinates.

    If no y are given the function returns (1 + x/c)**(-beta);
    If no w are given the function returns the normalized chi_square;
    if the w are given the function returns the chi_square weighted with the errors;
    '''
    if len(args) < 1:
        raise RuntimeError("No args passed to p_n_pow_const, at least the x are required!")
    x = args[0]
    y = None
    w = None
    if len(args) == 2:
        y = args[1]
    if len(args) == 3:
        y = args[1]
        w = args[2]

    bb = pp[0]
    cc = pp[1]
    model =  (1. + x/cc)**(-bb)
    if y is None:
        return model
    elif w is None:
        return (((model - y)/model)**2.).sum()
    else:
        return (((model - y)/w)**2.).sum()


def Power_Growth(pp, *args):
    '''
    Power_Growth([A, m, s], , [y], [w])

    Evaluates y = A * x**m + s
    '''
    x = args[0]
    y = None
    w = None
    if len(args) == 2:
        y = args[1]
    if len(args) == 3:
        y = args[1]
        w = args[2]

    A = pp[0]
    m = pp[1]
    s = pp[2]

    model =  A*(x**m) + s
    if y is None:
        return model
    elif w is None:
        return (((model - y)/model)**2.).sum()
    else:
        return (((model - y)/w)**2.).sum()



########################################
########################################

def Smooth_Curve(x_in, y_in, w_in=None, binning='log', factor=1.5):
    '''
    Smoothing of a curve in selected bins.

    Smooth_Curve(x, y, w=None, binning='log', factor=1.5)

    w: np.array
        the weights to use to compute the average of a curve in a Delta x interval. If None is given
        they are set to w = np.ones(len(x))
    binning: str
        If binning is set to 'log' factor sets the number of bins as max(2, x.max()/x.min()/factor).
        If binning is set to 'lin' factor sets the number of bins as max(2, len(x)/factor).
    '''
    x = np.array(x_in, dtype=float)
    y = np.array(y_in, dtype=float)

    # Sorting the arrays for convenience...
    indxs = np.argsort(x)
    x, y = x[indxs], y[indxs]

    if w_in is None:
        w = np.ones(len(x), dtype=float)
    else:
        w = np.array(w_in, dtype=float)
        w = w[indxs]

    xmin, xmax = x[0], x[-1]
    x_bins = np.logspace(np.log10(xmin*.99999), np.log10(xmax*1.00001), max(2, np.ceil(xmax/float(xmin)/float(factor)))) if binning=='log'\
            else np.linspace(xmin*.99999, xmax*1.00001, max(4, int(np.ceil(len(x)/factor))))

    Xcenters = (x_bins[1:] + x_bins[:-1])/2.
    Ysum = np.zeros(len(Xcenters), dtype=float)
    Yden = np.zeros(len(Xcenters), dtype=float)

    bin_indx = 0
    for xi,yi,wi in zip(x,y,w):
        if xi >= x_bins[bin_indx+1]:
            bin_indx += 1
        Ysum[bin_indx] += yi*wi
        Yden[bin_indx] += wi

    Xcenters = Xcenters[Yden>.0]
    Ysum = Ysum[Yden>.0]
    Yden = Yden[Yden>.0]

    return Xcenters, Ysum/Yden


########################################
########################################


def GAUSS(p, x, d = None, w = None):
    C = p['cc'].value
    m = p['mm'].value
    s = p['ss'].value
    model = C*np.exp(-(x - m)**2./s)
    if d is None:
        return model
    if w is None:
        return model - d
    return (model - d)*w

def Log_Binning_noW(x, y, nb = None):
    x = np.array(x)
    y = np.array(y)

    if nb is None:
        nb  = np.ceil(x.shape[0]/3.)

    y = y[x != .0]
    x = x[x != .0]

    bx = np.logspace(np.log10(x.min()*.99), np.log10(x.max()*1.01), nb)
    BX = np.zeros(bx.shape[0] - 1, float)
    BY = np.zeros(bx.shape[0] - 1, float)
    CY = np.zeros(bx.shape[0] - 1, float)

    for bi in range(len(bx) - 1):
        BX[bi] = (bx[bi] + bx[bi + 1])/2.
    for (ii, xi) in enumerate(x):
        i = 0
        while not ( (xi >= bx[i]) & (xi < bx[i + 1]) ):
            i += 1
        BY[i] += y[ii]
        CY[i] += 1.

    BX = BX[np.nonzero(CY)]
    BY = BY[np.nonzero(CY)]
    CY = CY[np.nonzero(CY)]
    BY /= CY
    return {'x': BX, 'y': BY, 'c': CY}


def Log_Binning(x, y, w = None, nb = None):
    x = np.array(x)
    y = np.array(y)
    if w is None:
        w = np.ones(len(x), float)
    else:
        w = np.array(w)
    if nb is None:
        nb  = np.ceil(x.shape[0]/3.)
    w = w[x != .0]
    y = y[x != .0]
    x = x[x != .0]
    bx = np.logspace(np.log10(x.min()*.99), np.log10(x.max()*1.01), nb)
    BX = np.zeros(bx.shape[0] - 1, float)
    BY = np.zeros(bx.shape[0] - 1, float)
    CY = np.zeros(bx.shape[0] - 1, float)
    WW = np.zeros(bx.shape[0] - 1, float)
    for bi in range(len(bx) - 1):
        BX[bi] = (bx[bi] + bx[bi + 1])/2.
    for (ii, xi) in enumerate(x):
        i = 0
        while not ( (xi >= bx[i]) & (xi < bx[i + 1]) ):
            i += 1
        BY[i] += y[ii]
        WW[i] += w[ii]
        CY[i] += 1.
    BX = BX[np.nonzero(CY)]
    BY = BY[np.nonzero(CY)]
    WW = WW[np.nonzero(CY)]
    CY = CY[np.nonzero(CY)]
    BY /= CY
    return {'x': BX, 'y': BY, 'w': WW, 'c': CY}

def Lin_Log_Bins(start, stop, factor=1.5, firstWidth=None):
    assert(start <= stop)
    assert(factor > 1.)

    bmin = max(1., start)
    bmax = np.ceil(max(1., stop)) + .5

    if firstWidth is None:
        nb = int(np.ceil(np.log(bmax/bmin)/np.log(factor))) + 1
        b = np.logspace(np.log10(bmin), np.log10(bmax), nb)
        b_temp = [bmin-.5, np.ceil(b[1])+.5]
        step = b_temp[1] - b_temp[0]
    else:
        secondVal = bmin+firstWidth+.5
        b_temp = [bmin-.5, secondVal]
        nb = int(np.ceil(np.log(bmax/secondVal)/np.log(factor))) + 1
        b = np.logspace(np.log10(secondVal), np.log10(bmax), nb)
        step = np.ceil(b[1] - b[0])


    for bval in b[2:]:
        if ( (np.floor(bval) + .5 - b_temp[-1]) >= step ):
            b_temp.append(np.floor(bval)+.5)
            step = b_temp[-1] - b_temp[-2]

    if b_temp[-1] < bmax:
        b_temp.append(bmax)

    if start < 1:
        if firstWidth is None:
            b_temp = [-.5] + b_temp
        else:
            b_temp[0] = -.5

    b_temp = np.array(b_temp)

    return b_temp

def nestedBins(binning_scheme, Activities, Degrees, Entrances, act_bins_factor, deg_bins_factor, entr_bins_factor, firstEntranceWidth=None):
    '''
    Given the activities, the degrees and the entrances time (if needed), returns a dictionary
    of bins made like this:

    `
    Bins = {
            "b": [array],   # bins extremes of the first level;
            "v":
                {i: {                   # i is the i-th bin collector
                        "b": [array]    # bins of the second level for this subclass
                        "v":
                            {
                                ...     # same for level 3 (if present)
                            }
                    }
                }
            }
    `
    '''

    tmpActs = np.array(Activities)
    tmpDegs = np.array(Degrees)
    tmpEntr = np.array(Entrances)

    sys.stdout.write("Calculating the nodes bins in %d levels: %s\n" % (len(binning_scheme), binning_scheme))
    Bins = {}
    for index, what in enumerate(binning_scheme):
        if what == "a":
            tmp_vec = tmpActs
            tmp_step = act_bins_factor
        elif what == "k":
            tmp_vec = tmpDegs
            tmp_step = deg_bins_factor
        elif what == "e":
            tmp_vec = tmpEntr
            tmp_step = entr_bins_factor

        if index == 0:
            if len(tmp_vec) == 0:
                tmp_bins = np.array([-.5, .5])
            else:
                if what == "e":
                    tmp_bins = Lin_Log_Bins(tmp_vec.min(), tmp_vec.max()+1, tmp_step,\
                            firstWidth=firstEntranceWidth)
                else:
                    tmp_bins = Lin_Log_Bins(tmp_vec.min(), tmp_vec.max()+1, tmp_step)

            Bins["b"] = tmp_bins
            Bins["v"] = {i: {} for i in range(len(tmp_bins)-1)}

            # Label the copies...
            if what == "a":
                tmpActs = np.array([np.argmax(tmp_bins>b)-1 for b in tmp_vec])
            elif what == "k":
                tmpDegs = np.array([np.argmax(tmp_bins>b)-1 for b in tmp_vec])
            elif what == "e":
                tmpEntr = np.array([np.argmax(tmp_bins>b)-1 for b in tmp_vec])

        elif index == 1:
            if binning_scheme[0] == "a":
                tmp_vec_0 = tmpActs
            elif binning_scheme[0] == "k":
                tmp_vec_0 = tmpDegs
            elif binning_scheme[0] == "e":
                tmp_vec_0 = tmpEntr

            for bins_0, values_0 in Bins["v"].iteritems():
                indices = tmp_vec_0 == bins_0
                tmp_vals = tmp_vec[indices]

                if len(tmp_vals) == 0:
                    tmp_bins = np.array([-.5, .5])
                else:
                    tmp_bins = Lin_Log_Bins(tmp_vals.min(), tmp_vals.max()+1, tmp_step)

                values_0["b"] = tmp_bins
                values_0["v"] = {i: {} for i in range(len(tmp_bins)-1)}

                # Label the copies...
                if what == "a":
                    tmpActs[indices] =\
                            np.array([np.argmax(tmp_bins>b)-1 for b in tmp_vec[indices]])
                elif what == "k":
                    tmpDegs[indices] =\
                            np.array([np.argmax(tmp_bins>b)-1 for b in tmp_vec[indices]])
                elif what == "e":
                    tmpEntr[indices] =\
                            np.array([np.argmax(tmp_bins>b)-1 for b in tmp_vec[indices]])

        elif index == 2:
            if binning_scheme[0] == "a":
                tmp_vec_0 = tmpActs
            elif binning_scheme[0] == "k":
                tmp_vec_0 = tmpDegs
            elif binning_scheme[0] == "e":
                tmp_vec_0 = tmpEntr

            if binning_scheme[1] == "a":
                tmp_vec_1 = tmpActs
            elif binning_scheme[1] == "k":
                tmp_vec_1 = tmpDegs
            elif binning_scheme[1] == "e":
                tmp_vec_1 = tmpEntr

            for bins_0, values_0 in Bins["v"].iteritems():
                for bins_1, values_1 in values_0["v"].iteritems():
                    indices = np.where((tmp_vec_0 == bins_0) & (tmp_vec_1 == bins_1))
                    tmp_vals = tmp_vec[indices]
                    if len(tmp_vals) == 0:
                        tmp_bins = np.array([-.5, .5])
                    else:
                        tmp_bins = Lin_Log_Bins(tmp_vals.min(), tmp_vals.max()+1, tmp_step)

                    values_1["b"] = tmp_bins
                    values_1["v"] = {i: {} for i in range(len(tmp_bins)-1)}

        sys.stdout.write("Done level %d - %s out of %d...\n" % (index+1, what, len(binning_scheme)))
    return Bins
