#!/bin/bash

PYTHON_CMD=python2

DATFILE=$1
ODir=$2

Func_type="pow"
act_bins_classes=" 5 6 7 8"

LOG_ID=0
while [ -e "RunLog_Analysis_"$LOG_ID".txt" ]; do
    ((LOG_ID++));
done

RUNLOG=RunLog_Analysis_"$LOG_ID".txt
echo "Saving info in $RUNLOG ..."


# p_n old way...
$PYTHON_CMD plot-p_n_a.py $DATFILE $ODir $Func_type 1 0 2>&1 >> "$RUNLOG"


# p_n chi2 and heat map...
$PYTHON_CMD Betas_Chi2.py "$ODir"00/p_n_a/Betas_n_Curves.dat $ODir $Func_type 2>&1 >> "$RUNLOG"
BETA_OPT=$(cat $RUNLOG | grep "beta_opt_act=" | cut -d "=" -f 2)
echo "beta_opt = $BETA_OPT "
$PYTHON_CMD Heat_Map.py "$ODir"00/chi_square/Dat_Chi2.dat $ODir 2>&1 >> "$RUNLOG"



# <k(a,t)>, rhos and stuff...
$PYTHON_CMD P_akt.py $DATFILE $ODir $Func_type $BETA_OPT $act_bins_classes 2>&1 >> "$RUNLOG"

$PYTHON_CMD plot-rho_ak.py $DATFILE $ODir 1 $BETA_OPT 2>&1 >> "$RUNLOG"


