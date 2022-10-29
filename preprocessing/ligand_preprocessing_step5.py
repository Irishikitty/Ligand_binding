import pickle
import re
from collections import defaultdict
import os
import subprocess
import string
from tqdm import trange
import pickle
from collections import deque

letter_list = list(string.ascii_uppercase) + list(string.ascii_lowercase) + list('1234567890')
'''

all_ligand_atomtype_chain_axes_distill.pickle or all_ligand_atomtype_chain_axes.pickle
{ligand: [[pdbID, chainID, ligand atoms, axes], [pdbID, chainID, ligand atoms, axes]]}

pdb_chainid_list.pickle
[(pdbID, chainID), (pdbID, chainID)]

'''


def generate_ligand_pdbid_chainID_ligandatom_axes_list():

    all_ligand_atomtype_chain_axes_distill = pickle.load(open("all_ligand_atomtype_chain_axes_distill.pickle", "rb"))   # {ligand: [[pdbid, chainID, ligand_atom, axes]]}
    counter = 0
    ligand_pdbid_chainID_ligand_atom_axes_list = deque()
    for ligand, infos in all_ligand_atomtype_chain_axes_distill.items():
        for info in infos:
            ligand_pdbid_chainID_ligand_atom_axes_list.append(([ligand] + info))
            counter += 1
    print(counter)
    f = open("ligand_pdbid_chainID_ligand_atom_axes_list.pickle","wb")        #[[ligand, pdbid, chainID, ligand_atom, axes]]
    pickle.dump(ligand_pdbid_chainID_ligand_atom_axes_list,f)
    f.close()


if __name__=="__main__":

    pdb_chainid_list = pickle.load( open( "pdb_chainid_list.pickle", "rb" ))
    ligand_pdbid_chainID_ligand_atom_axes_list = pickle.load( open( "ligand_pdbid_chainID_ligand_atom_axes_list.pickle", "rb" ))

    sequence_data_dir = 'sequence_data_for_pdb/'
    pdb_seq_data_list = []
    # pdb_seq_dict_pdbID_chainID_grouped =  defaultdict(list)

    for index in trange(len(pdb_chainid_list)):
        pdbID, chainID = pdb_chainid_list[index]
        sequence_file = sequence_data_dir + pdbID + '.fasta'
        with open(sequence_file) as file:
            lines = file.readlines()
            for line_index, line in enumerate(lines):
                # print(line_index)
                if line.startswith(">"):
                    eles = line.strip().split("|")
                    temp_chainInfo = eles[1]
                    # chainInfo = re.sub(r"\[auth\s[A-Z]\]|Chain\s|Chains\s", '', temp_chainInfo)
                    chainInfo = re.sub(r"\w+\[auth\s|Chain\s|Chains\s|\]", '', temp_chainInfo)
                    chainInfo_list = chainInfo.split(", ")
                    try:
                        for id in chainInfo_list:
                            assert id in letter_list
                        if chainID in chainInfo_list:
                            head = line
                            sequence = lines[line_index + 1]
                            # pdb_seq_dict_pdbID_chainID_grouped[sequence].extend(chainInfo_list)
                            # print(sequence)
                            break
                    except:
                        print(pdbID, chainID)

        sequence_info = (pdbID, chainID, head, sequence)
        pdb_seq_data_list.append(sequence_info)

    f = open("pdb_seq_data_list.pickle","wb")
    pickle.dump(pdb_seq_data_list, f)
    f.close()








    # pair_index = [(i,j) for i in range(len(pdb_seq_data_list)) for j in range(i+1, len(pdb_seq_data_list))]
    # f = open("pdb_seq_fasta_pair.pickle","wb")
    # pickle.dump(pdb_seq_data_list, f)
    # f.close()





    # output = main()

    # pdbids_list = ['2R75', '6SJJ', '1BCH']
    # for index, pdbid in enumerate(pdbids_list):
    #     run_string = f"curl -o fasta_data_small/{pdbid}.fasta  https://www.rcsb.org/fasta/entry/{pdbid}/"
    #     tokens = run_string.split()
    #     subprocess.run(tokens)

    # pdb_chainid_list = [('6SJJ', 'A')]
    # for index in trange(len(pdb_chainid_list)):
    #     pdbID, chainID = pdb_chainid_list[index]
    #     sequence_file = 'fasta_data_small/' + pdbID + '.fasta'
    #     with open(sequence_file) as file:
    #         lines = file.readlines()
    #         for line_index, line in enumerate(lines):
    #             # print(line_index)
    #             if line.startswith(">"):
    #                 eles = line.strip().split("|")
    #                 temp_chainInfo = eles[1]
    #                 # chainInfo = re.sub(r"\[auth\s[A-Z]\]|Chain\s|Chains\s", '', temp_chainInfo)
    #                 # chainInfo = re.sub(r"auth\s|Chain\s|Chains\s", '', temp_chainInfo)
    #                 chainInfo = re.sub(r"\w+\[auth\s|\Chain\s|Chains\s|\]", '', temp_chainInfo)
    #                 chainInfo_list = chainInfo.split(", ")
    #                 if chainID in chainInfo_list:
    #                     head = line
    #                     sequence = lines[line_index + 1]
    #                     print(sequence)
    #                     break

    #
    #     pdbid, ligand_name, chainid, ligand_data, env_atoms_data = data
    #     print(pdbid, ligand_name, chainid)
    #     sequence_file = sequence_data_dir + pdbid + '.fasta'
    #     with open(sequence_file) as file:
    #         lines = file.readlines()
    #         for line_index, line in enumerate(lines):
    #             print(line_index)
    #             if line.startswith(">"):
    #                 eles = line.strip().split("|")
    #                 temp_chainInfo = eles[1]
    #                 chainInfo = re.sub(r"\[auth\s[A-Z]\]|Chain\s|Chains\s", '', temp_chainInfo)
    #                 chainInfo_list = chainInfo.split(", ")
    #                 print(chainInfo_list)
    #                 if chainid in chainInfo_list:
    #                     head = line
    #                     sequence = lines[line_index + 1]
    #                     print(sequence)
    #                     break
    #     new_data = (pdbid, ligand_name, chainid, ligand_data, env_atoms_data, head, sequence)
    #     ligand_env_coords_data_seq.append(new_data)
    #
    #
    #
    #
    #
    # for indexA, data_seqA in enumerate(ligand_env_coords_data_seq):
    #     pdbidA, _, chainidA, _, _, headA, seqA = data_seqA
    #     # print(pdbidA, chainidA)
    #     for indexB, data_seqB in enumerate(ligand_env_coords_data_seq):
    #         pdbidB, _, chainidB, _, _, headB, seqB = data_seqB
    #         if indexA < indexB:
    #             with open('seq_data_for_comparison/' + pdbidA + '_' + chainidA + '-' + pdbidB + '_' + chainidB + '.fa', 'w') as file:
    #                 file.write(headA)
    #                 file.write(seqA)
    #                 file.write(headB)
    #                 file.write(seqB)












    # atom_types_dict = defaultdict(list)
    # for ligand, infos in all_ligand_atom_types.items():
    #
    #     print('=='*100)
    #     print(ligand)
    #     N = len(infos)
    #     for n in range(N):
    #         pdbid = infos[n][0]
    #         chainid = infos[n][1]
    #         print(len(infos[n][2]))
    #         assert len(infos[n][2]) == len(infos[n][3])
            # print(pdbid, chainid, infos[n][3])
    #
    # for k, v in file.items():
    #     print('==' * 100)
    #     print('current ligand :', k)
    #     try:
    #         ligand_name = k
    #         print('current ligand in try block :', ligand_name)
    #         ligand_length = ligand_len[k]
    #         count_ligands += 1
    #         N = len(v)
    #         for n in range(N):
    #             # ligand axes
    #             pdbid = v[n][0]
    #             chainid = v[n][1]
    #             ligand_coords = v[n][3]  # coords of ligand atoms
    #             if len(v[n][1]) == ligand_length:  # then no missing atoms
    #
    #                 # protein coords
    #                 file_name = '/Users/keqiaoli/Desktop/axial_attention/pdb_files_all_new/' + pdbid.lower() + '.pdb'
    #                 env_atom, env_coords, resName, channels = read_pdb(file_name)
    #                 ligand_center = np.mean(ligand_coords, 0)
    #                 dist = pairwise_distances(ligand_center.reshape(1, -1), env_coords)
    #                 idx = np.argsort(dist)[0]  # sorted index
    #
    #                 env_atom = [env_atom[i] for i in idx][:200]
    #                 resName = [resName[i] for i in idx][:200]
    #                 env_coords = [env_coords[i] for i in idx][:200]
    #                 channels = [channels[i] for i in idx][:200]
    #
    #                 # ligand_axes [22, x, y, z], atom_axes[atom_type[dict], ax, ay, az]
    #                 ligand_coords = [[22, *i] for i in ligand_coords]
    #                 env_coords = [[i, *j] for i, j in zip(channels, env_coords)]
    #                 data.append((pdbid, ligand_name, chainid, ligand_coords, env_coords))
    #
    #         if (count_ligands % 1000) == 0:
    #             print(time.time() - t1)
    #             print('Number of ligand left:', len(file) - count_ligands)
    #     except:
    #         print(ligand_name)
    #         ligand_name_cache.append(ligand_name)