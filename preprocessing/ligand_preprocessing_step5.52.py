import pickle
from tqdm import tqdm, trange
from collections import defaultdict

'''

pdb_chainid_list.pickle
[(pdbID, chainID), (pdbID, chainID)]
pdb_seq_data_list.pickle
[(pdbID, chainID, head, sequence), (pdbID, chainID, head, sequence)]

'''

if __name__=='__main__':



    seq_data_list = pickle.load(open("seq_data_list.pickle", "rb"))
    print(len(seq_data_list))

    seq_2_index = {seq: f'seq_{index}' for index, seq in enumerate(seq_data_list)}
    index_2_seq = {f'seq_{index}': seq for index, seq in enumerate(seq_data_list)}

    f = open("seq_2_index.pickle","wb")
    pickle.dump(seq_2_index, f)
    f.close()
    f = open("index_2_seq.pickle","wb")
    pickle.dump(index_2_seq, f)
    f.close()


    seq_pair_dict = defaultdict(list)
    for i in trange(len(seq_data_list)):
        if i == len(seq_data_list)-1:
            continue
        seq_pair_dict[seq_data_list[i]].extend([seq_data_list[j] for j in range(i+1, len(seq_data_list))])
    f = open("seq_pair_dict.pickle","wb")
    pickle.dump(seq_pair_dict, f)
    f.close()


    total = len([seqB for _, others in seq_pair_dict.items() for seqB in others])
    print(total)

    counter = 0
    stop_indicator = False
    sub_seqA_list = []
    for index, (seq, others) in enumerate(seq_pair_dict.items()):
        for seqB in others:
            counter += 1
            if counter > total/2:
                median_seq = seq
                stop_indicator = True
                break
        if stop_indicator:
            break
        print(index)

    print('final index is: ', index)
    sub_seqA_listA = list(seq_pair_dict.keys())[:index+1]
    sub_seqA_dictA = {seqA: seq_pair_dict[seqA] for seqA in sub_seqA_listA }



    f = open("sub_seqA_listA.pickle","wb")
    pickle.dump(sub_seqA_listA, f)
    f.close()
    f = open("sub_seqA_dictA.pickle","wb")
    pickle.dump(sub_seqA_dictA, f)
    f.close()


    sub_seqA_listB = list(seq_pair_dict.keys())[index + 1:]
    sub_seqA_dictB = {seqA: seq_pair_dict[seqA] for seqA in sub_seqA_listB }

    f = open("sub_seqA_listB.pickle","wb")
    pickle.dump(sub_seqA_listB, f)
    f.close()
    f = open("sub_seqA_dictB.pickle","wb")
    pickle.dump(sub_seqA_dictB, f)
    f.close()


