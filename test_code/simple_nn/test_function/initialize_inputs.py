import sys
sys.path.append('../../')
from simple_nn_v2 import simple_nn
from simple_nn_v2 import init_inputs

logfile = open('LOG', 'w', 10)

inputs = init_inputs.initialize_inputs('input.yaml', logfile)
print(inputs)
