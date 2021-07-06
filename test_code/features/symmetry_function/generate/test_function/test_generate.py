import os

#Check ase version available#
import ase

#Set yor python directory or command
python_dir='python3'
#Set python test files to run
test_list=['parse_symmetry_function_parameters.py',
           'get_structure_coordination_info.py',           
           'get_atom_types_info.py',
           'initialize_result.py',
           'initialize_symmetry_function_variables.py'
          ]



print('Start testing symmetry_function.generate.')

if ase.__version__ == '3.21.1':
    print("ase version 3.21.1 is not available due to io.read. try other version")
    os.abort()
else:
    print("ase version OK")

FILE = '../../../../test_data/SiO2/OUTCAR_comp'
print(f'Copy data file {FILE} to current directory.\n')
os.system(f'cp {FILE} ./')

for number, test in enumerate(test_list):
    print(f'TEST {number+1} : {test} start.')
    success_info = eval(r'os.system("{0} {1}")'.format(python_dir,test))
    if success_info == 0:
        print(f'TEST {number+1} : {test} done.\n')
    else:
        print(f'TEST {number+1} : in {test} error occured. aborting.')
        os.abort()
