# pyCoNet

__*A suite to analyze Time-Varying-Networks within the Activity-Driven framework featuring
reinforcement process and burstiness.*__

This project is a **HUGE** mess and still a work in progress. The working parts, so far, are the
importer tool and some of the automatic plot utilities. However many scripts still have parameters
to be set by the user within the source code and many consistency checks are still missing. I am
working to make everything comfortable to use but so far __use it at your own risk__!

## Installation

Just clone/download/copy/whatever\_you\_want this repository in your path or in your working folder
and make sure to have the dependencies installed.

### Dependencies:

- numpy;
- matplotlib;
- seaborn.

### Input file format

The program expects as input a list of (or a single) files contained in a given directory (clean it from hidden files and subdirectories as well);
The file can be either a zipped file or a regular text file.
Each line must contain the following information:

    Caller_ID   Called_ID   Company_Caller   Company_Called   [time]

where `Calle*_ID` are the IDs of caller (the node engaging the interaction) and called node (the node receiving the interaction), and then `Company_Calle*` specifies whether or not we have to take into account the caller or called nodes in the averages evaluation. Finally, `time` (optional) can tell the program the reference time of the simulation or of the dataset used as clock (may be the event number or the date or whatever you want).


###Importing and analyzing a sequence of contacts

Everything is done through the `Network_Importer` function whose keyword arguments may be tuned for a very customized importing procedure.
From the documentation:

```python
Analyze and import a sequence of contacts.

Usage:
Network_Importer(IDir, ODir, SocTHr=0, step_by_step=True, n_t_smpl=10, TIME_events=True, **kwargs)

Parameters
----------
IDir: string
   path to the input directory;
ODir: string
    path to the output directory;
SOC_THR: int
    Social threshold, i.e. only edges with w_ij >= SocThr and w_ji >= SocThr will be kept. Default is 0.
step_by_step: bool
    whether or not to use the step by step importing rather than the aggregated view per file. Default is True.
n_t_smpl: int
    number of log-spaced times to analyze: (if no time is specified in the input files time will be measured
    according to time_events). Default is 10.
time_events: bool
    if no time is specified in the input files we will assume an event-measured time, if set to `False` we will
    measure time in file number. Default is True.
 **kwargs *:
       binning: str
            ['act'], 'entr': whether to bin the nodes depending on their total activity (default) or by their
            entrance time.
       bins_factor: float
            The factor of the activity/entrance time and degree log-bins. Default is 1.25 .
       entr_min: int
            The first bin of the entrance time goes from 1 to entr_min. Default is 10000.
       entr_bin_fact: float
            The factor for the entrance time binning. Default is 2.
       tini: int
            The analysis will begin from the tini-th file in the folder. Default is `0`.
       tfin: int
            The analysis will end at the tfin-th file in the folder. Default is `Number_of_files`.
       zipped_f: bool
            Whether or not the files in the folder are zipped. Default is `True`.
       clust_perc: float
            The percentual of nodes on which to compute the clustering coefficient. Default is 0.2 (i.e. 20%).
       starting_time: int or float
            The first time at which we analyze the P(a,k,t) and the clustering coefficient. Default is 5 if time is measured in files number,
            .05 if event time/time specified (i.e. we start after 5% of the total time has passed).

The file expects an input file structured as follows:
CALLER_ID \t CALLED_ID \t COMPANY_CLR \t COMPANY_CLD \t TIME_event \n

The last column is optional and sets the time of reference of the event. If not given we will
assume the following time:
    - each event is at time t_i=number_of_lines if the number of files < 2*n_t_smpl;
    - each event inside a single file is at time #num_of_file;
```

The function expects an input file structured as follows:

    CALLER_ID \t CALLED_ID \t COMPANY_CLR \t COMPANY_CLD \t TIME_event \n

The last column is optional and sets the time of reference of the event. If not given we will
assume the following time:
- each event is at time $t_i=\rm{line number}$ if `time_events` is set to `True`;
- each event inside a single file is at time `num_of_file` if `time_events` is set to `False` (integrated version);


##Output structure
The output of the importing function is stored in a single zipped file containing the data-structure resulting from the analysis.
The output has the following structure:

```python
DATA = {
    'PAR': {
        'Got_Time': whether_or_not_time_was_specified_in_the_input_files,
        'Time_Events': whether_or_not_the_time_was_measured_as_events,
        'step_by_step': step_by_step_or_not,
        'n_t_smpl': number_of_timed_analysis,
        'entr_bin_fact': entrance_bins_factor,
        'bins_factor': activity_and_degree_bins_factor,
        'clust_perc': percentage_of_nodes_over_which_clustering_is_computed,
        'starting_time': entrance time for the first bin that goes from [ev=1 to starting_time),
        },
    'TOT': {
        'Stats': {
            'TOT': {'Nodes': N, 'CompanyNodes': Nc, 'Edges': E, 'Events': Evs},
            'SOC': {'Threshold': SocThr, 'Nodes': N, 'CompanyNodes': Nc, 'Edges': E, 'Events': Evs},
        },
        'N_A_K': {
            primary_bin: {deg_bin: number_of_nodes_in_this_bin},
        },
        'Vectors': {
            # Note that all of these vectors are coherent, so all the features of each node get
            # saved in the same location.
            'aa':   Activities_as_number_of_events_engaged,
            'kk':   Cumulative_degree,
            'kin':  Cumulative_In_Degree,
            'kout': Cumulative_Out_Degree,
            'win':  Cumulative_In_Weight,
            'wout': Cumulative_Out_Weight,
            't0':   Entrance time (in events) of node,
        },

        ###############################################
        # All the subsequent vectors have the same length and each position refers to a specifical
        # analysis of the evolving system.
        'fn_t': [times_as_file_number at all the points of the TVec analysis],
        're_t': [times_as_reference_time at all the points of the TVec analysis (practically a copy of TVec)],
        'ev_t': [times_in_events_number at all the points of the TVec analysis],
        'TVec': [the_times_at_which_analysis_occured (practically a copy of 're_t')],

        'avgk_t': [Overall_average_degree_at_time_t],
        'avgclust_t': [Overall_average_clustering_coefficient_at_time_t],
        'edge_t': [Active_edges_at_time_t],
        'Node_t': [Active_nodes_at_time_t],
        ###############################################
    },
    'ACT': {
            'Bins': {
                'aa': [Lin_Log_Binned_edges_of_activity_counted_as_number_of_events],
                'kk': {
                    act_bin: [Lin_Log_Binned_edges_of_degree_for_the_activity_class],
                },
                't0': [Lin_Log_Binned_edges_of_entrance_time_counted_as_number_of_events],
            },
            'k_a_t': {
                act_bin: {index_of_time_in_TVec: avg_degree_for_act_bin if one_active_node else None},
            },
            'P_akt': {
                act_bin: {
                    time_index_in_TVec: {
                    'k': [k for k in range(k_min_for_act_bin, k_max_for_act_bin + 1)],
                    'n': [frequency_of_the_corresponding_k_value_at_this_time_for_act_bin],
                    },
                },
            },
            'P_N_A': {
                act_bin: {
                    deg_bin: {
                        k: {
                            's_new': num_of_events_toward_new_nodes,
                            's_eve': tot_num_of_events_for_bin,
                        },
                    },
                },
            },
            'Pw': {
                act_bin: {
                    'in':  {'b': [win for win in range(min(wins), max(wins)+1)], 'w': [frequencies]},
                    'out': {'b': [wout for wout in range(min(wouts), max(wouts)+1)], 'w': [frequencies]},
                },
            },
            'Pk': {
                act_bin: {
                    'tot': {'b': [k for k in range(min(ktots), max(ktots)+1)],   'k': [frequencies]},
                    'in':  {'b': [ki for ki in range(min(kins), max(kins)+1)],   'k': [frequencies]},
                    'out': {'b': [ko for ko in range(min(kouts), max(kouts)+1)], 'k': [frequencies]},
                },
            },
    },
}
```
so that, for example, we can evaluate the $p(k)$ reinforcement process for a selected nodes bin by simply doing:
```python
act_bin = 4
deg_bin = 3
if act_bin in DATA['ACT']['P_N_A'] and deg_bin in DATA['ACT']['P_N_A'][act_bin]:
    p_n_raw = DATA['ACT']['P_N_A'][act_bin][deg_bin]

    degrees = np.array([k for k, diz in sorted(p_n_raw.items()) if diz['s_eve'] > 0])
    p_n = np.array([p_n_raw[k]['s_new']/float(p_n_raw[k]['s_eve']) for k in degrees])

    plt.loglog(degrees, p_n, '.--', ms=6, alpha=.8, lw=2)
```
or we can plot the rescaled $P(a,k,t)$ by typing:
```python
act_bin = 8

# The reference time we want to select (in this case events)...
ref_time = DATA['TOT']['ev_t']

if act_bin in DATA['ACT']['P_akt']:
    Pakt = DATA['ACT']['P_akt'][act_bin]

    for t_index in range(len(ref_time)):
        ks = np.array(Pakt[t_index]['k'])
        ns = np.array(Pakt[t_index]['n'])

        avg_k = (ks*ns).sum()/ns.sum()

        XXX = (ks - avg_k)/np.sqrt(avg_k)
        YYY = ns*np.sqrt(avg_k)

        if t_index < len(ref_time)-1: # Short times get plotted lighter than the long time limit curve
            plt.plot(XXX, YYY, '--*', lw=1.2, alpha=.6, ms=5, label=r'$t=%.01e$' % ref_time[t_index])
        else:
            plt.plot(XXX, YYY, '-o',  lw=2.,  alpha=.9, ms=8, label=r'$t=%.02e$' % ref_time[t_index])

```

## Automatic $p(n)$ plot and analysis

    plotting_scripts/plot-p_n_a.py ../DATASETS/00/data/DATFile.dat OUTDIR 'func_type' FlagFit Flag_Compact

that returns the data file *Betas_n_Curves.dat* that can be passed to *Betas_Chi2.py*:

    python Betas_Chi2.py ../DATASET/p_n_a/Betas_n_Curves.dat ../out/dir/ fit_func:{'pow','exp'}



##The last commands in turn gives you the *Dat_Chi2.dat* that can be passed to *Heat_Map.py*:
    python2 Heat_Map.py ../DATASET/00/chi_square/Dat_Chi2.dat ../path/to/Odir/

###Note that both *Betas_Chi2.py* and *Heat_Map.py* have parameters to be set in the header!



##And for the $P(a,k,t)$:
    python2 P_akt.py ../DATASET/00/data/DATA.dat ../output_folder/ fit_func_type('pow' - 'exp') Beta act_bin and NOBins


Then, have a look at all the plotting scripts and at the Notebooks for further happy-plotting.

##The automator is the script:##


