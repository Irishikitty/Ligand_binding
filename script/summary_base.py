
# Read PDB ligands and protein env ---------------------
## 1. Read ligands,
#  all_ligand_atom_type: dict, {ligand_name: [[pdbid],[ligand_atom_seq],[axes]]}
#  all_ligand_length: dict, {ligand_name: [[pdbid],[len(ligand_atom_seq)]]}
from script.utils import *
import os
import time
import pickle
os.getcwd()

# DIR -------------------------------------------------
target_ligands_file = './script/saved_ligands_new.txt'
all_ligand_data_file = './script/ligand_dict_new.json'
pdb_files_dir = './data/'
pdb_files = os.listdir(pdb_files_dir)

# Load MOAD base to read PDB ---------------------------
all_ligand_atom_type = defaultdict(list)
all_ligand_length = defaultdict(list)
with open(all_ligand_data_file, 'r') as f:
    all_ligand_data = json.load(f)
pdbfiles = set()
for v in all_ligand_data.values():
    temp = set([k for i,j,k in list(v)])
    pdbfiles = set.union(pdbfiles, temp)
pdbfiles = [i.lower() for i in pdbfiles]
','.join(pdbfiles)

def main():
    pdbid_download = []
    t1 = time.time()
    with open(target_ligands_file, 'r') as file:
        for count, ligand in enumerate(file):
            # Read ligand ---------
            ligand_name = ligand.strip().strip('\"')
            pdb_info_list = all_ligand_data[ligand_name]
            ## 1-molecule ligand
            if len(ligand.strip().split(" "))==1:
                try:
                    for pdb_info in pdb_info_list:
                        chain_name, position, pdbid = pdb_info
                        target_pdb_file = pdb_files_dir + pdbid.lower() + '.pdb'
                        read_results = read_ligand_simple(target_pdb_file, ligand_name, chain_name, position)
                        if read_results not in [1,2]:
                            ligand_atom = [i for (i,j) in read_results]
                            axes = [j for (i,j) in read_results]
                            all_ligand_atom_type[ligand_name].append([pdbid, ligand_atom, axes])
                            all_ligand_length[ligand_name].append([pdbid, len(ligand_atom)])
                except:
                    # Print PDB id if reading issue happened
                    print('print pdbbid',pdbid)
                    pdbid_download.append(pdbid)

            ## multi-molecule ligand
            else:
                assert len(ligand.strip().split(" ")) > 1
                try:
                    for pdb_info in pdb_info_list:
                        chain_name, position, pdbid = pdb_info
                        target_pdb_file = pdb_files_dir + pdbid.lower() + '.pdb'
                        read_results = read_ligand_complex(target_pdb_file, ligand_name, chain_name, position)
                        if read_results != 1:
                            ligand_atom = [i for (i, j) in read_results]
                            axes = [j for (i, j) in read_results]
                            all_ligand_atom_type[ligand_name].append([pdbid, ligand_atom, axes])
                            all_ligand_length[ligand_name].append([pdbid,len(ligand_atom)])
                except:
                    print('print pdbid',pdbid)
                    pdbid_download.append(pdbid)

                if (count % 2000) == 0:
                    print('time:', time.time() - t1)
    print(','.join(pdbid_download))
    print('Finished!')

    # create a binary pickle file
    f = open("all_ligand_atom_type.pkl","wb")
    pickle.dump(all_ligand_atom_type,f)
    f.close()

if __name__ == '__main__':
    main()

