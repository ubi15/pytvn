#!/bin/bash

PYTHON_CMD=python2

IMPORT_CMD=Network_Importer_step-by-step.py

ANALYSIS_CMD=./Auto_Analysis.sh

IDir=$1
ODir=$2
ST=$3
Times=$4

LOG_ID=0
while [ -e "RunLog_Import_"$LOG_ID".txt" ]; do
    ((LOG_ID++));
done

RUNLOG=RunLog_Import_"$LOG_ID".txt
echo "Saving info in $RUNLOG ..."


# Importing...
$PYTHON_CMD $IMPORT_CMD $IDir $ODir $ST $Times 2>&1 >> "$RUNLOG"
DATFILE=$(cat $RUNLOG | grep "Data saved in:" | cut -d ":" -f 2)

echo "Imported in $DATFILE , now the analysis..."

$ANALYSIS_CMD "$DATFILE" "$ODir"


