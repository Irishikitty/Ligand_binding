import math
import os
import pickle
import re
import subprocess
from collections import defaultdict
import multiprocess as mp
import time
from tqdm import tqdm
from multiprocessing import get_context

'''

pdb_chainid_list.pickle
[(pdbID, chainID), (pdbID, chainID)]
pdb_seq_data_list.pickle
[(pdbID, chainID, head, sequence), (pdbID, chainID, head, sequence)]

'''


def write_fasta_files(seq1andseq2, seq1_index, seq2_index, dir):
    seq1, seq2 = seq1andseq2
    with open(dir + '/' + seq1_index + '-' + seq2_index + '.fa','w') as file:
        output = "\n".join(['>' + seq1_index, seq1, '>' + seq2_index, seq2])
        file.write(output)
    # run_string = f"/data/keqiaoli/protein_ligand_binding_air/data_preprocessing/run_tcoffee.sh {output_dir} {seq_2_index[seqA]} zipped_files"
    # tokens = run_string.split()
    # subprocess.run(tokens)

def calc_chunksize(n_workers, len_iterable, factor=4):
    """
    Calculate chunksize argument for Pool-methods.
    Resembles source-code within `multiprocessing.pool.Pool._map_async`.
    """
    chunksize, extra = divmod(len_iterable, n_workers * factor)
    if extra:
        chunksize += 1
    return chunksize


if __name__ == "__main__":

    step = 100000
    seq_2_index = pickle.load(open('seq_2_index.pickle', 'rb'))
    index_2_seq = pickle.load(open('index_2_seq.pickle', 'rb'))
    seq_pair_dict = pickle.load(open('seq_pair_dict.pickle', 'rb'))
    # sub_seqA_dict = pickle.load(open('sub_seqA_dict' + ID + '.pickle', 'rb'))



    seq_pair_dict_equal = {}
    counter = 0
    seqs_list = []
    for seqA, others in seq_pair_dict.items():
        for seqB in others:
            counter += 1
            if counter % step == 0:
                seqs_list.append((seqA, seqB))
                seq_pair_dict_equal[counter] = seqs_list
                seqs_list = []
            else:
                seqs_list.append((seqA, seqB))
    seq_pair_dict_equal[counter] = seqs_list


    max_number1 = len([pair for counter, seqs in seq_pair_dict_equal.items() for pair in seqs])
    max_number2 = len([(seqA, seqB) for seqA, others in seq_pair_dict.items() for seqB in others])
    print(max_number1, max_number2)
    # print(f'Generating {max_number1} files ...')
    # output_dir ='seq_data_for_tcoffee'
    # os.mkdir(output_dir)


    for counter, seqs in tqdm(seq_pair_dict_equal.items()):

        res = counter % step
        if not res:
            count = int(counter / step)
        else:
            count = int(counter / step) + 1
        print(counter, count)
        # sub_output_dir = output_dir + '/' + f'seq_mix_{count}'
        # os.mkdir(sub_output_dir)

        # with get_context("fork").Pool(processes=15) as pool:
        #
        #     # chunk_size = calc_chunksize(180, len(seq_pair_dict[seqA]))
        #     chunk_size = math.ceil(len(seqs) / 15)
        #
        #     pool.starmap_async(write_fasta_files,
        #                        [(pair,
        #                        seq_2_index[pair[0]],
        #                        seq_2_index[pair[1]],
        #                        sub_output_dir)
        #                       for pair in seqs], chunk_size)
        #
        #     pool.close()
        #     pool.join()

        # run_string = f"/data/keqiaoli/protein_ligand_binding_air/data_preprocessing/zip_file.sh {output_dir} seq_mix_{count} zipped_files1"
        # tokens = run_string.split()
        # subprocess.run(tokens)
    print('Finished !')



