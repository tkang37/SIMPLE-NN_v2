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
    4. initialize result and symmetry function variables

"""
# 1. parsing params_XX
symf_params_set = descriptor._parsing_symf_params()

# 2. load snapshot from FILE
from ase import io
FILE = '../test_data/generate/OUTCAR_2'
snapshots = io.read(FILE, index='::', format='vasp-out')
snapshot = snapshots[0]
structure_tags = ['None', 'Data1', 'Data2', 'Data3']
structure_weights = [1, 1, 3, 3]

# 3. extract type_idx, atom_num from _get_structure_info()
cart_p, scale_p, cell_p, atom_num, atom_i, atom_i_p, type_num, type_idx = descriptor._get_structrue_info(snapshot, structure_tags, structure_weights)

# 4. initialize result and symmetry function variables
idx = 1
result = descriptor._init_result(type_num, structure_tags, structure_weights, idx, atom_i)
jtem = 'Sb'
cal_num, cal_atoms_p, x, dx, da, x_p, dx_p, da_p = descriptor._init_sf_variables(type_idx, jtem, symf_params_set, atom_num, mpi_range = None )

# 5. calculate symmetry function
from simple_nn_v2.features.symmetry_function._libsymf import lib, ffi
errno = lib.calculate_sf(cell_p, cart_p, scale_p, atom_i_p, atom_num,\
        cal_atoms_p, cal_num, symf_params_set[jtem]['ip'],\
        symf_params_set[jtem]['dp'], symf_params_set[jtem]['num'],\
        x_p, dx_p, da_p)

""" Main test code

    Test _set_result()
    1. check if "jtem" atom number is correct
    2. check if "jtem" atom idx is correct
    3. check if 'x', 'dx', 'da' is initialze to 0
    4. check if 'x_p', 'dx_p', 'da_p' is initialize to 0

"""

result = descriptor._set_result(result, x, dx, da, type_num, jtem, symf_params_set, atom_num)
print(result.keys())
print 'N: ', result['N']
print 'tot_num: ', result['tot_num']
print 'partition: ', result['partition']
print 'struct_type: ', result['struct_type']
print 'struct_weight: ', result['struct_weight']
prev=0
end=0
for elem in result['N']:
    end += result['N'][elem]
    print 'result["N"][%s] atom_idx: '%elem, result['atom_idx'][prev:end], len(result['atom_idx'][prev:end])
    prev += result['N'][elem]
print 'partition_%s: '%jtem, result['partition_%s'%jtem]
print(result['x'])
