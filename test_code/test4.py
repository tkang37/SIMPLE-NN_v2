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

""" Previous setting before test code

    1. parsing params_XX
    2. load snapshot from FILE
    3. extract type_idx, atom_num from _get_structure_info()

"""
symf_params_set = descriptor._parsing_symf_params()

from ase import io
FILE = '../test_data/generate/OUTCAR_2'
snapshots = io.read(FILE, index='::', format='vasp-out')
snapshot = snapshots[0]
structure_tags = ['Data1', 'Data2', 'Data3']
structure_weights = [1, 3, 3]
cart_p, scale_p, cell_p, atom_num, atom_i, atom_i_p, type_num, type_idx = descriptor._get_structrue_info(snapshot, structure_tags, structure_weights)


""" Main test code

    Test _init_sf_variables()
    1. check if "jtem" atom number is correct
    2. check if "jtem" atom idx is correct
    3. check if 'x', 'dx', 'da' is initialze to 0
    4. check if 'x_p', 'dx_p', 'da_p' is initialize to 0

"""
jtem = 'Te'
cal_num, cal_atoms_p, x, dx, da, x_p, dx_p, da_p = descriptor._init_sf_variables(type_idx, jtem, symf_params_set, atom_num, mpi_range = None )

print 'cal_num: ', cal_num
print('cal_atom_p: '),
for i in range(cal_num):
    print(cal_atoms_p[i]),
print
print 'x', x
print 'dx', dx
print 'da', da
for i in range(cal_num):
    for j in range(symf_params_set[jtem]['num']):
        if x_p[i][j] != 0 :
            print '"x_p" %sth atom, %sth value is not zero'%(i, j)
for i in range(cal_num):
    for j in range(symf_params_set[jtem]['num']*atom_num*3):
        if dx_p[i][j] != 0 :
            print '"dx_p" %sth atom, %sth value is not zero'%(i, j)
for i in range(cal_num):
    for j in range(symf_params_set[jtem]['num']*3*6):
        if da_p[i][j] != 0 :
            print '"da_p" %sth atom, %sth value is not zero'%(i, j)
