import pickle
from tqdm import tqdm
import string
letter_list = list(string.ascii_uppercase)
if __name__=='__main__':

    ''' 
    pdb_seq_data_list = pickle.load(open("pdb_seq_data_list.pickle", "rb"))
    print(len(pdb_seq_data_list))
    seq_data_list = list(set([ele[3].strip('\n') for ele in pdb_seq_data_list]))
    f = open("seq_data_list.pickle", "wb")
    pickle.dump(seq_data_list, f)
    f.close()
    '''


    # pdb_seq_data_list = pickle.load(open("pdb_seq_data_list.pickle","rb"))     #[(pdbID, chainID, head, sequence), (pdbID, chainID, head, sequence)]
    # all_ligand_pdbid_chain_atomtype_axes_pos_distill = pickle.load(open("all_ligand_pdbid_chain_atomtype_axes_pos_distill.pickle", "rb"))   # {ligand: [[pdbid, chainID, ligand_atom, axes, pos]]}
    #
    # for ligand, infos in tqdm(all_ligand_pdbid_chain_atomtype_axes_pos_distill.items()):
    #     for info in infos:
    #         pdbid, chainID, _, _, _ = info
    #         for ele in pdb_seq_data_list:
    #             if pdbid == ele[0] and chainID == ele[1]:
    #                 info.append(ele[3].strip('\n'))
    # # create a binary pickle file
    # f = open("all_ligand_pdbid_chain_atomtype_axes_pos_seq.pickle","wb")
    # pickle.dump(all_ligand_pdbid_chain_atomtype_axes_pos_distill,f)
    # f.close()

    all_ligand_pdbid_chain_atomtype_axes_pos_seq = pickle.load(open("all_ligand_pdbid_chain_atomtype_axes_pos_seq.pickle", "rb"))
    sequence_2_ligandInfo = {}
    for ligand, infos in tqdm(all_ligand_pdbid_chain_atomtype_axes_pos_seq.items()):
        for info in infos:
            pdbid, chainID, ligand_atoms, axes, pos, seq = info
            if seq not in sequence_2_ligandInfo.keys():
                sequence_2_ligandInfo[seq] = [[pdbid, chainID, ligand_atoms, axes, pos, ligand]]
            else:
                sequence_2_ligandInfo[seq].append([pdbid, chainID, ligand_atoms, axes, pos, ligand])

    f = open("sequence_2_ligandInfo.pickle","wb")
    pickle.dump(sequence_2_ligandInfo,f)
    f.close()
    print('Finished!')
