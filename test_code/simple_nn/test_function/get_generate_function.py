import sys
sys.path.append('../../')

from simple_nn_v2 import simple_nn

logfile = open('LOG', 'w', 10)

print('Get symmetry_function generator')
generate = simple_nn.get_generate_function(logfile, descriptor_type='symmetry_function')
print(generate)

print('Get unimplemented generator')
generate = simple_nn.get_generate_function(logfile, descriptor_type='others')
print(generate)
