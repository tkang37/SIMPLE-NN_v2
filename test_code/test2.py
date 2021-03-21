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

    1. load snapshot from FILE

"""
from ase import io
FILE = '../test_data/generate/OUTCAR_1'
snapshots = io.read(FILE, index='::', format='vasp-out', force_consistent=True)
snapshot = snapshots[0]

""" Main test code

    Test _get_structure_info()
    1. check if "atom_num" is total atom number
    2. check Cartesian coordination in "cart_p" (first 5, last 5 coordination)
    3. check Fractional coordination in "cart_p" (first 5, last 5 coordination)
    4. check lattice parameter
    5. check if atom_i_p value is equal atom_i
    6. check "type_num" has correct element types and atom number for each elements
    7. check "type_idx" has correct atom index 

"""
import numpy as np
cart_p, scale_p, cell_p, atom_num, atom_i, atom_i_p, type_num, type_idx = descriptor._get_structrue_info(snapshot, structure_tags=None, structure_weights=None)
cart=np.copy(snapshot.get_positions(wrap=True), order='C')
scale = np.copy(snapshot.get_scaled_positions(), order='C')

print('1. check if "atom_num" is total atom number')
print('atom num: %s\n'%atom_num)

print('2. check Cartesian coordination in "cart_p" (first 5, last 5 coordination)')
print('Cartesian coordination')
for i in range(5):
    print 'cart: ', cart[i][0], cart[i][1], cart[i][2]
    print 'cart_p: ', cart_p[i][0], cart_p[i][1], cart_p[i][2]
print('...')
for i in range(len(cart_p)-5, len(cart_p)):
    print 'cart: ', cart[i][0], cart[i][1], cart[i][2]
    print 'cart_p: ', cart_p[i][0], cart_p[i][1], cart_p[i][2]

print('\n3. check Fractional coordination in "cart_p" (first 5, last 5 coordination)')
print('Fractional coordination')
for i in range(5):
    print 'scale: ', scale[i][0], scale[i][1], scale[i][2]
    print 'scale_p: ', scale_p[i][0], scale_p[i][1], scale_p[i][2]
print('...')
for i in range(len(scale_p)-5, len(scale_p)):
    print 'scale: ', scale[i][0], scale[i][1], scale[i][2]
    print 'scale_p: ', scale_p[i][0], scale_p[i][1], scale_p[i][2]

print('\n4. check lattice parameter')
print('Lattice parameter')
for i in range(3):
    print(cell_p[i][0], cell_p[i][1], cell_p[i][2])

print('\n5. check if atom_i_p value is equal atom_i')
print('atom_i: %s'%atom_i)
for i in range(atom_num):
    if atom_i_p[i] != atom_i[i]:
        print('%sth atom has different value'%i)

print('\n6. check "type_num" has correct element types and atom number for each elements')
print('type_num: %s'%type_num)

print('\n7. check "type_idx" has correct atom index ')
print('type_idx: %s'%type_idx)
