import subprocess
import json
'''
Download sequence data

all_ligand_atomtype_chain_axes_distill.pickle or all_ligand_atomtype_chain_axes.pickle
{ligand: [[pdbID, chainID, ligand atoms, axes], [pdbID, chainID, ligand atoms, axes]]}

'''





if __name__=="__main__":

    import os
    import pickle

    pdb_files_dir = os.getcwd() + '/pdb_files_all_new/'
    file = pickle.load(open("all_ligand_pdbid_chain_atomtype_axes_pos_distill.pickle", "rb"))
    pdbid_list = [ele[0] for _, value in file.items() for ele in value]

    pdbids_list_new = list(set(pdbid_list))
    print(len(pdbids_list_new))
    current_dir = os.getcwd()
    output_dir = current_dir + '/sequence_data_for_pdb'
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    for index, pdbid in enumerate(pdbids_list_new):
        run_string = f"curl -o {output_dir}/{pdbid}.fasta  https://www.rcsb.org/fasta/entry/{pdbid}/"
        tokens = run_string.split()
        subprocess.run(tokens)