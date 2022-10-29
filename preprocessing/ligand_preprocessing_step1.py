import os

import numpy as np
import pandas as pd
import sys
import json

def list_files_recursive(input_path):
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


def flatten(t):
    return [item for sublist in t for item in sublist]



if __name__=="__main__":
    import subprocess

    # subprocess.run('./run_try.sh')
    # target_ligands = list(pd.read_csv('all_valid_ligands.csv')['x'])
    output_filename = "/Users/keqiaoli/Desktop/protein_ligand_binding_air/data_preprocessing/ligand_chain_position_pdbid_dict.json"
    input_paths = ['/Users/keqiaoli/Desktop/Selected_Bindingdata']
    file_list = list()
    for input_path in input_paths:
        file_list.extend(list_files_recursive(input_path))
    # print(file_list)
    if '/Users/keqiaoli/Desktop/Selected_Bindingdata/.DS_Store' in file_list:
        file_list.remove('/Users/keqiaoli/Desktop/Selected_Bindingdata/.DS_Store')
    if '/Users/keqiaoli/Desktop/Selected_Bindingdata/.Rhistory' in file_list:
        file_list.remove('/Users/keqiaoli/Desktop/Selected_Bindingdata/.Rhistory')
    # print(len(file_list))
    all_ligand_data = {}
    for file in file_list:
        # print('=='*100)
        # print(file)
        df_data = pd.read_csv(file)
        pdbid = df_data.columns.values.tolist()[2]
        # print(df_data.iloc[:,3])
        rown, coln = df_data.shape
        for i in range(rown):
            assert df_data.iloc[i,3] in set(["invalid", "valid", "Part of Protein"])
        df_data_small = df_data[df_data.iloc[:,3] != "invalid"]

        ligand_candidate_list  = list(df_data_small.iloc[:,2])
        # print(ligand_candidate_list)
        for ligand in ligand_candidate_list:
            ligand_name  = ligand.split(":")[0].strip()
            chain_name = ligand.split(":")[1].strip()
            position = ligand.split(":")[2].strip()
            # print(ligand_name)
            if ligand_name in all_ligand_data.keys():
                # print("duplicate ligand found !!!")
                all_ligand_data[ligand_name].append([chain_name, position, pdbid])
            else:
                all_ligand_data[ligand_name] = [[chain_name, position, pdbid]]
            # print(ligand_name, chain_name, position, pdbid)
    # print(len(all_ligand_data.keys()))
    # print(all_ligand_data)
    with open(output_filename, 'w') as file:
        file.write(json.dumps(all_ligand_data, indent = 4))
    # for key in all_ligand_data.keys():
    #     print(key)
