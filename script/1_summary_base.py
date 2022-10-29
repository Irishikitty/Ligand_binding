
# Read PDB ligands and protein env ---------------------
## 1. Read ligands,
#  all_ligand_atom_type: dict, {ligand_name: [[pdbid],[ligand_atom_seq],[axes]]}
#  all_ligand_length: dict, {ligand_name: [[pdbid],[len(ligand_atom_seq)]]}
# import os
# os.chdir('/Users/menghanlin/Desktop/Ligand_binding/Ligand_binding')

from script.utils import *
import os
import time
import pickle
os.getcwd()

# DIR -------------------------------------------------
target_ligands_file = './script/collection/saved_ligands_new.txt'
all_ligand_data_file = './script/collection/ligand_dict_new.json'
pdb_files_dir = './data_pdb/'
pdb_files = os.listdir(pdb_files_dir)

# Load MOAD base to read PDB ---------------------------
all_ligand_atom_type = defaultdict(list)
all_ligand_length = defaultdict(list)
with open(all_ligand_data_file, 'r') as f:
    all_ligand_data = json.load(f)
# pdbfiles = set()
# for v in all_ligand_data.values():
#     temp = set([k for i,j,k in list(v)])
#     pdbfiles = set.union(pdbfiles, temp)
# pdbfiles = [i.lower() for i in pdbfiles]
# ','.join(pdbfiles)

def main():
    with open(target_ligands_file, 'r') as file:
        for count, ligand in enumerate(file):
            # Read ligand ---------
            # ligand_name = ligand.strip().strip('\"')
            ligand_name = 'CU'
            pdb_info_list = all_ligand_data[ligand_name]
            ## 1-molecule ligand
            # if len(ligand.strip().split(" "))==1:
            for pdb_info in pdb_info_list:
                chain_name, position, pdbid = pdb_info
                # if pdbid == '1AND':
                #     print('NOTE!!!')
                #     break
                    
                target_pdb_file = pdb_files_dir + pdbid + '.pdb'
                read_results = read_ligand_simple(target_pdb_file, ligand_name, chain_name, position)
                if read_results not in [1,2]:
                    ligand_atom = [i for (i,j) in read_results]
                    axes = [j for (i,j) in read_results]
                    all_ligand_atom_type[ligand_name].append([pdbid, ligand_atom, axes])
                    

    print('Finished!')

    # create a binary pickle file
    f = open("all_ligand_atom_type.pkl","wb")
    pickle.dump(all_ligand_atom_type,f)
    f.close()

if __name__ == '__main__':
    main()

