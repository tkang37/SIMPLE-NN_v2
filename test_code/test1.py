import sys
sys.path.append('../')
from simple_nn_v2 import Simple_nn
from simple_nn_v2.features.symmetry_function import Symmetry_function

# Minimum Setting for Testing Symmetry_function methods
# Initialize input file, set Simple_nn object as parent of Symmetry_function object
model = Simple_nn('input.yaml', descriptor=Symmetry_function())
descriptor = Symmetry_function()
descriptor.parent = model
descriptor.set_inputs()



""" Main test code

Test _parsing_symf_params()
1. Check if 'num' for each elements
2. Check key 'total', 'i', 'd' 'ip', 'id' values for each elements

"""

symf_params_set = descriptor._parsing_symf_params()
print(symf_params_set.keys())

# Check keys
# Check key 'num'
# Check type of key 'dp', 'ip'
for elem in symf_params_set.keys():
    print(elem)
    print(symf_params_set[elem].keys())
    print(symf_params_set[elem]['num'])
    print(symf_params_set['Ge']['dp'])
    print(symf_params_set['Ge']['ip'])
    print


# Check ['total'] values
# Check ['i'] values
# Check ['d'] values
for elem in symf_params_set.keys():
    print(elem)
    f=open('params_%s'%elem,'r')
    lines=f.readlines()
    f.close()

    for i, line in enumerate(lines):
        vals = line.split()
        for j in range(len(vals)):
            if float(vals[j]) != symf_params_set[elem]['total'][i][j]:
                print('key: total elem: ', elem, '  ', i+1,'th symf, ',j+1, 'th value')

            if j<3:
                if float(vals[j]) != symf_params_set[elem]['i'][i][j]:
                    print('key: i  elem: ', elem, '  ', i+1,'th symf, ',j+1, 'th value')
                if float(vals[j]) != symf_params_set[elem]['ip'][i][j]:
                    print('key: ip  elem: ', elem, '  ', i+1,'th symf, ',j+1, 'th value')
            elif j>=3:
                if float(vals[j]) != symf_params_set[elem]['d'][i][j-3]:
                    print('key: d  elem: ', elem, '  ',i+1,'th symf, ',j+1, 'th value')
                if float(vals[j]) != symf_params_set[elem]['dp'][i][j-3]:
                    print('key: dp  elem: ', elem, '  ',i+1,'th symf, ',j+1, 'th value')
    print

