import os
from collections import deque, defaultdict
import numpy as np
from copy import copy
import re


amio_acids_list = ['ARG', 'HIS', 'LYS', 'ASP', 'GLU', 'SER', 'THR', 'ASN', 'GLN', 'CYS', 'GLY', 'PRO', 'ALA', 'VAL',
                   'ILE', 'LEU', 'MET', 'PHE', 'TYR', 'TRP']


def list_files_recursive(input_path):
    '''
    List all pdb files under current location

    :param input_path: '.'
    :return: a list of file names ['./data/xxxx.pdb','./data/xxxx.pdb']
    '''

    file_list = list()
    dir_list = list()
    if os.path.isfile(input_path):
        file_list.append(input_path)
    elif os.path.isdir(input_path):
        dir_list.append(input_path)
    else:
        raise RuntimeError("Input path must be a file or directory: " + input_path)
    while len(dir_list) > 0:
        dir_name = dir_list.pop()
        # print("Processing directory " + dir_name)
        dir = os.listdir(dir_name)
        for item in dir:
            input_filename = dir_name
            if not input_filename.endswith("/"):
                input_filename += "/"
            input_filename += item
            #print("Checking item " + input_filename)
            if os.path.isfile(input_filename):
                file_list.append(input_filename)
            elif os.path.isdir(input_filename):
                dir_list.append(input_filename)
    return file_list

# ==============================================================================

def read_ligand_simple(file_name, name, chain, pos):
    '''
    Read 1-molecule ligand from pdb

    param (file_name, name, chain, pos): './data/xxxx.pdb', ligand name, chain, molecule position
    return:
        1: ligand in MOAD is not consistent with the one in pdb file (same chain ID)
        2: ligand is not found
        list of set: (atomSymbol, [1.2, 2.3, 3.4])
    '''
    # if check_models(file_name):
    multi_atoms_ligand = False
    with open(file_name) as f:
        lines_total = f.readlines()
        result = check_models(file_name)
        data_list = []
        if result:
            for (start, end) in result:
                model_lines = lines_total[start: end]
                data = get_ligand_simple_data(model_lines, name, chain, pos, temperature = True)
                if not isinstance(data, int):
                    if len(data) > 1:
                        multi_atoms_ligand = True
                    data_list.append(data)
            if multi_atoms_ligand:
                T_list = [[T for (_, _, T) in data] for data in data_list]
                index = np.argmin(np.mean(T_list, axis=1))
                data = [data_list[index][i][:-1] for i in range(len(data_list[index]))]
            else:
                T_list = [T for data in data_list for (_, _, T) in data]
                index = np.argmin(T_list)
                data = [data_list[index][i][:-1] for i in range(len(data_list[index]))]
            # if multi_atoms_ligand:
            #     T_list = [T for data in data_list for (_, _, T) in data]
            #     index = np.argmmin(np.mean(T_list, axis = ))
            # data = data_list[index]
        else:
            data = get_ligand_simple_data(lines_total, name, chain, pos)
        return data

# ==============================================================================


def get_ligand_simple_data(lines, name, chain, pos, temperature=False):
    data = []
    for line in lines:
        altLoc = line[16]
        if line[0:6] == 'HETATM' and (altLoc == ' ' or altLoc == 'A'):
            ligand_name = line[17:20].strip()
            chainID, position, atomSym = line[21], line[22:26].strip(), line[76:78].strip()  # atom type
            x, y, z = line[30:38].strip(), line[38:46].strip(), line[46:54].strip()
            T = line[60:66].strip()
            if chainID == chain and position == pos:
                if ligand_name != name:  # ???
                    return 1
                if atomSym != 'H':
                    if temperature:
                        data.append((atomSym, [float(i) for i in [x, y, z]], float(T)))
                    else:
                        data.append((atomSym, [float(i) for i in [x, y, z]]))
    if data == []:
        return 2
    return data

# ==============================================================================

def f_read_pdb_line(idx, lines, temp_factor):
    '''
    :param idx:
    :param lines:
    :return:
    '''
    temperature = None
    line = lines[idx]
    protein_ligand_prefix = line[0:6]
    ligand_name = line[17:20].strip()
    altLoc = line[16]
    chainID, position = line[21], line[22:26].strip()
    atomSym = line[76:78].strip()
    x,y,z = line[30:38].strip(), line[38:46].strip(), line[46:54].strip()
    if temp_factor:
        temperature = line[60:66].strip()

    return protein_ligand_prefix, ligand_name, chainID, position, altLoc, atomSym, (x, y, z), temperature

# ==============================================================================

def read_ligand_complex(file_name, name, chain, pos):
    '''
    Read multi-molecule ligand from pdb

    :param file_name: './abcd.pdb'
    :param name: ligand name 'NGA NAG'
    :param chain: chain 'A','B',...
    :param pos: position '1','2',...

    :return:
        1
        (list(set(data)), data): (unique atom type), (ligand atoms)
    '''

    tmp_name_list = name.split(' ')  # ['ABC', 'BCD', 'EFG', 'FGH']
    numLigands = len(tmp_name_list)
    sub_ligand_name_list = deque(tmp_name_list)
    data_list = []
    with open(file_name) as f:
        lines_total = f.readlines()
        result = check_models(file_name)
        if result:
            for (start, end) in result:
                model_lines = lines_total[start: end]
                data = get_ligand_complex_data(model_lines, copy(sub_ligand_name_list), chain, pos, temperature = True)
                data_list.append(data)

            T_list = [[T for (_, _, T) in ele] for ele in data_list]
            index = np.argmin(np.mean(T_list, axis=1))
            data = [data_list[index][i][:-1] for i in range(len(data_list[index]))]
        else:
            data = get_ligand_complex_data(lines_total, sub_ligand_name_list, chain, pos)
    return data





# ==============================================================================


def get_ligand_complex_data(lines, name_list, chain, pos, temperature = False, incomplete_indicator = False, first_ligand_is_found = False):
    data = []
    current_index = 0
    while name_list:
        target_ligand = name_list.popleft()
        if first_ligand_is_found:
            for index in range(current_index, len(lines)):
                protein_ligand_prefix, ligand_name, chainID, position, altLoc, atomSym, axes, T = f_read_pdb_line(
                    index, lines, temperature)

                if protein_ligand_prefix in set(['HETATM', 'ATOM  ']) and chainID == chain and position == pos:

                    next_protein_ligand_prefix, \
                    next_ligand_name, \
                    next_chainID, \
                    next_position, \
                    next_altLoc, \
                    _, \
                    next_axes, _ = f_read_pdb_line(index + 1, lines, temperature)

                    next_next_protein_ligand_prefix, \
                    next_next_ligand_name, \
                    next_next_chainID, \
                    next_next_position, \
                    next_next_altLoc, \
                    _, \
                    next_next_axes, _ = f_read_pdb_line(index + 2, lines, temperature)

                    if (altLoc == ' ' or altLoc == 'A') and atomSym != 'H':  # skip alternate position:
                        if ligand_name == target_ligand:
                            # first_ligand_is_found = True
                            # index_list.append(index)
                            if temperature:
                                assert T is not None
                                data.append((atomSym, [float(i) for i in axes], float(T)))
                            else:
                                assert T is None
                                data.append((atomSym, [float(i) for i in axes]))
                        else:
                            incomplete_indicator = True
                            break

                    if (next_position != position or (next_protein_ligand_prefix not in set(['HETATM', 'ATOM  '])) \
                        or next_chainID != chainID) and (
                            (next_next_position != position or next_next_chainID != chainID)
                            or next_next_protein_ligand_prefix not in set(['HETATM', 'ATOM  '])):
                        current_index = index + 1
                        break
                else:
                    incomplete_indicator = True
                    break

        else:
            for index in range(current_index, len(lines)):
                protein_ligand_prefix, ligand_name, chainID, position, altLoc, atomSym, axes, T = f_read_pdb_line(index,
                                                                                                               lines, temperature)

                if protein_ligand_prefix in set(['HETATM', 'ATOM  ']) and chainID == chain and position == pos:

                    next_protein_ligand_prefix, \
                    next_ligand_name, \
                    next_chainID, \
                    next_position, \
                    next_altLoc, \
                    _, \
                    next_axes, _ = f_read_pdb_line(index + 1, lines, temperature)

                    next_next_protein_ligand_prefix, \
                    next_next_ligand_name, \
                    next_next_chainID, \
                    next_next_position, \
                    next_next_altLoc, \
                    _, \
                    next_next_axes, _ = f_read_pdb_line(index + 2, lines, temperature)

                    if (altLoc == ' ' or altLoc == 'A') and atomSym != 'H':  # skip alternate position:
                        if ligand_name == target_ligand:
                            first_ligand_is_found = True
                            if temperature:
                                data.append((atomSym, [float(i) for i in axes], float(T)))
                            else:
                                data.append((atomSym, [float(i) for i in axes]))
                        else:
                            incomplete_indicator = True
                            break

                    if (next_position != position or (next_protein_ligand_prefix not in set(['HETATM', 'ATOM  '])) \
                        or next_chainID != chainID) and (
                            (next_next_position != position or next_next_chainID != chainID)
                            or next_next_protein_ligand_prefix not in set(['HETATM', 'ATOM  '])):
                        current_index = index + 1
                        break

        if incomplete_indicator:
            break

        pos = str(int(pos) + 1)

    if not incomplete_indicator and not name_list:
        return data
    else:
        return 1

# ==============================================================================

def check_HETATM(filename):
    with open(filename) as f:
        lines = f.readlines()
        for i in range(len(lines)):
            if lines[i][0:6] == 'HETATM' and lines[i+1][0:6] != 'HETATM' and lines[i+2][0:6] == 'HETATM':
                print(filename)

# ==============================================================================

def check_models(filename):
    range_list = []
    with open(filename)as f:
        lines = f.readlines()
        for line_number in range(len(lines)):
            if lines[line_number][0:5] == 'MODEL':
                start = line_number
            if lines[line_number][0:6] == 'ENDMDL':
                end = line_number
                range_list.append((start, end))
    return range_list

# ==============================================================================

def label_atom(resname,atom):
    atom_label={
    'ARG':{'N':17,'CA':18,'C':19,'O':20,'CB':4,'CG':4,'CD':4,'NE':11,'CZ':1,'NH1':11,'NH2':11},
    'HIS':{'N':17,'CA':18,'C':19,'O':20,'CB':4,'CG':6,'CD2':6,'CE1':6,'ND1':8,'NE2':8},
    'LYS':{'N':17,'CA':18,'C':19,'O':20,'CB':4,'CG':4,'CD':4,'CE':4,'NZ':10},
    'ASP':{'N':17,'CA':18,'C':19,'O':20,'CB':4,'CG':4,'OD1':15,'OD2':15},
    'GLU':{'N':17,'CA':18,'C':19,'O':20,'CB':4,'CG':4,'CD':2,'OE1':15,'OE2':15},
    'SER':{'N':17,'CA':18,'C':19,'O':20,'CB':4,'OG':13},
    'THR':{'N':17,'CA':18,'C':19,'O':20,'CB':3,'OG1':13,'CG2':5},
    'ASN':{'N':17,'CA':18,'C':19,'O':20,'CB':4,'CG':1,'OD1':14,'ND2':9},
    'GLN':{'N':17,'CA':18,'C':19,'O':20,'CB':4,'CG':4,'CD':1,'OE1':14,'NE2':9},
    'CYS':{'N':17,'CA':18,'C':19,'O':20,'CB':4,'SG':16},
    'GLY':{'N':17,'CA':18,'C':19,'O':20},
    'PRO':{'N':17,'CA':18,'C':19,'O':20,'CB':4,'CG':4,'CD':4},
    'ALA':{'N':17,'CA':18,'C':19,'O':20,'CB':5},
    'VAL':{'N':17,'CA':18,'C':19,'O':20,'CB':3,'CG1':5,'CG2':5},
    'ILE':{'N':17,'CA':18,'C':19,'O':20,'CB':3,'CG1':4,'CG2':5,'CD1':9},
    'LEU':{'N':17,'CA':18,'C':19,'O':20,'CB':4,'CG':3,'CD1':5,'CD2':5},
    'MET':{'N':17,'CA':18,'C':19,'O':20,'CB':4,'CG':4,'SD':16,'CE':5},
    'PHE':{'N':17,'CA':18,'C':19,'O':20,'CB':4,'CG':6,'CD1':6,'CD2':6,'CE1':6,'CE2':6,'CZ':6},
    'TYR':{'N':17,'CA':18,'C':19,'O':20,'CB':4,'CG':6,'CD1':6,'CD2':6,'CE1':6,'CE2':6,'CZ':6,'OH':13},
    'TRP':{'N':17,'CA':18,'C':19,'O':20,'CB':4,'CG':6,'CD1':6,'CD2':6,'NE1':7,'CE2':6,'CE3':6,'CZ2':6,'CZ3':6,'CH2':6}
    }
    if atom == 'OXT':
        return 21   # define the extra oxygen atom OXT on the terminal carboxyl group as 21 instead of 27 (changed on March 19, 2018)
    else:
        return atom_label[resname][atom]

# ==============================================================================

def check_peptide(ligand_name):

    ligand_name_list = ligand_name.split(' ')  # ['ABC', 'BCD', 'EFG', 'FGH']
    return True if len(set(ligand_name_list) & set(amio_acids_list)) else False

# ==============================================================================

# def get_chainIDs(all_lines):
#     list_chainID = []
#     for line in all_lines:
#         if line[0:6] == 'ATOM  ':
#             chainID = line[21]
#             if chainID not in list_chainID:
#                 list_chainID.append(chainID)
#             else:
#                 continue
#     return list_chainID

def get_chainIDs(all_lines):

    removed_chainID_list = []
    ser_chainID_dict = {}
    for line in all_lines:
        if line[0:6] == 'SEQRES':
            serNum = line[8:10].strip()
            chainID = line[11].strip()
            resNames = re.split(r"\s+", line[19:70].strip())
            if not all([resName in amio_acids_list for resName in resNames]):
                removed_chainID_list.append(chainID)

            numRes = line[13:17].strip()

            if chainID not in ser_chainID_dict.keys():
                ser_chainID_dict[chainID] = numRes
            else:
                continue

    # ser_chainID_dict_output = {k: v for k, v in ser_chainID_dict.items() if len(v) > 1}
    list_chainID = [k for k, v in ser_chainID_dict.items() if  int(v) >= 20]
    # list_chainID = [k for k, v in ser_chainID_dict.items() if (k not in removed_chainID_list and int(v) >= 20)]
    return list_chainID

# ==============================================================================

def check_missing_residues(all_lines):
    start = None
    end = None
    start_index_found = False
    missing_residue = {}
    for numLine, line in enumerate(all_lines):
        if line[0:27] == 'REMARK 465   M RES C SSSEQI':
            start = numLine
            start_index_found = True
        if line[0:10] != 'REMARK 465' and line[10: 26] == '                ' and start_index_found:
            end = numLine
            break
    if start and end:
        target_lines = all_lines[start+1: end]
        for line in target_lines:
            if line[19].strip() not in missing_residue.keys():
                missing_residue[line[19].strip()] = [line[15: 18].strip()]
            else:
                missing_residue[line[19].strip()].append(line[15: 18].strip())

    return missing_residue



# ==============================================================================

def read_pdb(file_name, model_index = None, result = None):

    pdb_for_each_model = {}

    with open(file_name) as f:
        lines_total = f.readlines()
        chainID_list = get_chainIDs(lines_total)
        if chainID_list:
            pdb_atoms_chainID = {}
            # result = check_models(file_name)
            for chainID in chainID_list:
                if result:
                    count = 1
                    for (start, end) in result:
                        model_lines = lines_total[start: end]
                        atomTypes, atomCoords, _ = get_env_atoms(model_lines, chainID)
                        if not atomTypes and not atomCoords:     # ignore this chain for the current model index
                            count += 1
                            continue
                        assert atomTypes != [] and atomCoords != []
                        pdb_for_each_model[f'model {count}'] = (atomTypes, atomCoords)
                        count += 1

                    if f'model {model_index}' in pdb_for_each_model.keys():
                        atomTypes, atomCoords = pdb_for_each_model[f'model {model_index}']
                    else:
                        continue
                else:
                    atomTypes, atomCoords, _ = get_env_atoms(lines_total, chainID)
                    if not atomTypes and not atomCoords:
                        continue
                #     return None
                pdb_atoms_chainID[chainID] = (atomTypes, atomCoords)
            return pdb_atoms_chainID
        else:
            return None

    # return atomTypes, atomCoords



# ==============================================================================

def get_model_index(file_name, name, chain, pos, result):

    tmp_name_list = name.split(' ')  # ['ABC', 'BCD', 'EFG', 'FGH']
    numLigands = len(tmp_name_list)
    sub_ligand_name_list = deque(tmp_name_list)
    data_list = []
    multi_atoms_ligand = False
    with open(file_name) as f:
        lines_total = f.readlines()
        # result = check_models(file_name)
        for (start, end) in result:
            model_lines = lines_total[start: end]
            if numLigands > 1:
                data = get_ligand_complex_data(model_lines, copy(sub_ligand_name_list), chain, pos, temperature = True)
                data_list.append(data)
                multi_atoms_ligand = True
            else:
                assert numLigands == 1
                data = get_ligand_simple_data(model_lines, name, chain, pos, temperature = True)
                if not isinstance(data, int):
                    if len(data) > 1:
                        multi_atoms_ligand = True
                    data_list.append(data)

        if multi_atoms_ligand:
            T_list = [[T for (_, _, T) in data] for data in data_list]
            index = np.argmin(np.mean(T_list, axis=1))

        else:
            T_list = [T for data in data_list for (_, _, T) in data]
            index = np.argmin(T_list)

    return index + 1


# # ==============================================================================
#
# def read_pdb(file_name):
#
#     data = []
#     with open (file_name) as f:
#         for line in f:
#             if line[0:6]=='ATOM  ':
#                 atomName = line[12:16].strip()
#                 x = float(line[30:38].strip())
#                 y = float(line[38:46].strip())
#                 z = float(line[46:54].strip())
#                 altLoc = line[16]
#                 atomSym = line[76:78].strip()
#                 resName = line[17:20]
#                 if atomSym != 'H':
#                     # The corresponding atom channel, given residual name and atom type
#                     label = label_atom(resName, atomName)
#                     if altLoc == ' ' or altLoc == 'A':  # skip alternate position
#                         data.append([label,x,y,z])
#     return data

# ==============================================================================


def get_env_atoms(lines, chain):
    atomNames = []
    resNames = []
    coords = []
    chain_start_end_dict = get_ter_index(lines)
    start, end = chain_start_end_dict[chain]
    for index, line in enumerate(lines[start:end]):
        if line[0:6] == 'ATOM  ':
            assert line[21] == chain
            atomName = line[12:16].strip()
            x = float(line[30:38].strip())
            y = float(line[38:46].strip())
            z = float(line[46:54].strip())
            altLoc = line[16]
            atomSym = line[76:78].strip()
            resName = line[17:20]

            if (atomSym != 'H') and (altLoc == ' ' or altLoc == 'A') and line[21] == chain:
                try:
                    label = label_atom(resName, atomName)
                    atomNames.append(atomName)
                    resNames.append(resName)
                    coords.append([label, x, y, z])
                except KeyError:
                    return None, None, None

    return atomNames, coords, resNames



# ==============================================================================

def get_ter_index(lines):
    ter_index_list = [(line[21], index) for index, line in enumerate(lines) if line[0:6] == 'TER   ' ]

    chain_start_end_dict = {}
    start = 0
    for i in range(len(ter_index_list)):
        chain, end = ter_index_list[i]
        chain_start_end_dict[chain] = [start, end]
        start = end+1
    return chain_start_end_dict




if __name__=="__main__":

    # for index, i in enumerate(['a','b', 'c', 'd', 'e']):
    #     print(index)
    #     if index != 2:
    #         if i == 'c':
    #             print(index)
    #         else:
    #             continue
    #     else:
    #         print('next')
    #
    # counter = 1
    # for ele in ['1','b', '3', 'd', 'e']:
    #     if ele not in ['a','b', 'c', 'd', 'e']:
    #         counter += 1
    #         continue
    #     print(counter)
    #     counter +=1


    # import pickle
    # dataset = pickle.load(open("dataset.pickle","rb"))
    # print(dataset)
    # length_list = {'train': [], 'validation': [], 'test': []}
    # for key, value in dataset.items():
    #     for data in value:
    #         ligand = data[-1]
    #         output = check_peptide(ligand)
    #         if output:
    #             if ligand == 'GLU' and data[0] == '2XXR' and data[1] == 'A' and data[-2] == '900':
    #                 for i in range(len(data[-3])):
    #                     print(data[-3][i])
    #                 # print(data[-3])
    #                 print(ligand, data[0], data[1], data[-2])


            # if len(ligand_name.split(' ')) > 1:
            #     print(ligand_name, data[-2])
            # print(len(ligand_name.split(' ')))

    pdbfile_name = ['2gvj.pdb', '1n51.pdb', '5cad.pdb', '2h06.pdb']
    for file in pdbfile_name:
        with open(file) as f:
            lines = f.readlines()
            chainIDs = get_chainIDs(lines)
            # ter_index_list, chain_loc_dict = get_ter_index(lines)
            atomNames, coords, resNames = get_env_atoms(lines, 'A')
            # missing_residue = check_missing_residues(lines)
            print(ter_index_list)
            print(chain_loc_dict)






            # length_list[key].append(len(data[2]))
    # data_list = [ [('N', [11.171, -21.985, 32.03], 22.64)], [('C', [10.904, -23.33, 32.043], 25.44)],
    #  [('N', [11.37, -23.981, 33.171], 22.29)], [('C', [12.081, -23.4, 34.243], 21.97)],
    #  [('C', [12.641, -22.098, 33.962], 21.36)], [('C', [12.059, -21.339, 32.985], 21.43)] ]
    #
    #
    #
    #
    # data = data_list[0]
    # print(len(data_list))
    #
    # multi_atoms_ligand = False
    # if len(data) > 1:
    #     multi_atoms_ligand = True
    # # data_list.append(data)
    # if multi_atoms_ligand:
    #     T_list = [[T for (_, _, T) in data] for data in data_list]
    #     index = np.argmin(np.mean(T_list, axis = 0))
    #     data = data_list[index]
    # else:
    #     T_list = [T  for data in data_list for (_, _, T) in data]
    #     index = np.argmin(T_list)
    #     data = data_list[index]

    # output1 = read_ligand_simple('2q4v.pdb', 'ACO HOH', 'A', '306')
    # output1 = read_ligand_simple('2q4v.pdb', 'HOH', 'A', '307')
    # index = get_model_index('2q4v.pdb', 'HOH', 'A', '307')
    # print(output1)
    # output11 = [output1[i][:-1] for i in range(len(output1))]
    # print(output11)

    # new_tuple_1 = tuple(item for item in my_tuple if item != 'two')

    # output1 = read_ligand_simple('2cl6.pdb', 'CAG', 'X', '167')
    # print(output1)
    #
    # output2 = read_ligand_complex('1n51.pdb','01B PRO PRO ALA NH2', 'B', 1)
    # print(output2)



    # output3 = read_ligand_simple('6n2e.pdb', 'PLM', 'D', '201')
    # print(output3)
    # print(len(output3))
    # output2 = read_ligand_simple('2icy.pdb', 'UPG', 'A', '901')
    # print(output2)
    # atomsyms1 = [atomsym for (atomsym, axe) in output1]
    # # atomsyms2 = [atomsym for (atomsym, axe) in output2]
    # print(atomsyms1)
    # print(atomsyms2)
    # print(output1)
    # print(len(atomsyms))
    # filenames = ['1c2h.pdb', '2q4h.pdb']
    # for filename in filenames:
    #     result = check_models(filename)
    #     if  result:
    #         print('multiple models', filename)
    #     else:
    #         print('single model', filename)
    # print(result)
    # print(len(result))


    # print(len(output))
