python_command=python	# In case you use another one...
Input_folder=../relative/path/to_input/files/ # WITH TRAILING SLASH!!!

# 1 month
$python_command Network_Importer_step-by-step_netlibs.py $Input_folder MPC 0 10 30 10 0 15
$python_command Network_Importer_step-by-step_netlibs.py $Input_folder MPC 0 10 30 10 30 45
$python_command Network_Importer_step-by-step_netlibs.py $Input_folder MPC 0 10 30 10 60 75
$python_command Network_Importer_step-by-step_netlibs.py $Input_folder MPC 0 10 30 10 80 95

# 2 months
$python_command Network_Importer_step-by-step_netlibs.py $Input_folder MPC 0 10 30 10 0 29
$python_command Network_Importer_step-by-step_netlibs.py $Input_folder MPC 0 10 30 10 20 49
$python_command Network_Importer_step-by-step_netlibs.py $Input_folder MPC 0 10 30 10 40 69
$python_command Network_Importer_step-by-step_netlibs.py $Input_folder MPC 0 10 30 10 60 89

# 4 months
$python_command Network_Importer_step-by-step_netlibs.py $Input_folder MPC 0 10 30 10 0 57
$python_command Network_Importer_step-by-step_netlibs.py $Input_folder MPC 0 10 30 10 40 97
