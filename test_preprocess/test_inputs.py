import sys
import os
import atexit
sys.path.append('../')

#DataGenerator.py -> datagenerator.py in utils
#DataGenerator Class -> Datagenerator


import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


#os.system('python libsymf_builder.py')

from simple_nn_v2 import run
from simple_nn_v2.init_inputs import  initialize_inputs


# Test function dictionary from input.yaml 
logfile = sys.stdout
logfile = open('LOG', 'w', 10)
inputs = initialize_inputs('input.yaml', logfile)
print('KEY of input.yaml')
for key in inputs.keys():
    print(key,'   :   ',inputs[key])
print('')
print('KEY of symmeytry_function')
for key in inputs['symmetry_function'].keys():
    print(key,'   :   ',inputs['symmetry_function'][key])
print('')
print('KEY of neural_network')
for key in inputs['neural_network'].keys():
    print(key,'   :   ',inputs['neural_network'][key])
