import math
import os
import pickle
from collections import defaultdict
from multiprocessing import get_context
from tqdm import tqdm


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

def read_filename_list(filename):
    # print(filename)
    with open(filename) as file:
        lines = file.readlines()
        # lines = [line.split('/')[1] for line in open(filename)]
    return lines


def read_fasta_output(filename):

    seqA_index, seqB_index = filename.split('/')[2].split('.')[0].split('-')
    with open(filename) as file:
        lines = file.readlines()
        score = float(lines[4][16:21])
        result = (seqA_index, seqB_index, score)
    return result

def log_result(result):
    print("Succesfully get callback! With result: ", result)



if __name__ == "__main__":

    dir_name = 'similarity_score/sim_filename_list'

    file_name_list = list_files_recursive(dir_name)
    filename_dict = {}
    for filename in tqdm(file_name_list):
        if filename.endswith('.txt'):
            lines = read_filename_list(filename)
            filename_dict[
                filename.split('/')[0] + '/' + filename.split('/')[-1].split('.')[0].strip('_filename')] = lines
            # filename_dict[filename.split('/')[0] + '/'] = lines
    similarity_score_dict = {}
    # score_list = []

    for pre_dir, filenames in tqdm(filename_dict.items()):

        print(f'reading all files now ...')
        with get_context("fork").Pool(processes=180) as pool:

            chunk_size = math.ceil(len(filenames) / 180)
            print('starting multiprocess now')
            result = pool.starmap_async(read_fasta_output, [('/'.join([pre_dir, file.strip()]),) for file in filenames],
                               chunk_size, callback=log_result)
            print('saving multiprocess result')
            similarity_score_dict[pre_dir] = {(seqA_index, seqB_index) : score for (seqA_index, seqB_index, score) in result.get()}
            # similarity_score_dict = {(seqA_index, seqB_index) : score for (seqA_index, seqB_index, score) in result.get()}
            pool.close()
            pool.join()

    print(similarity_score_dict.keys())
    print('=='*50)
    # print(scores)
    #
    f = open("similarity_score_dict_aws.pickle","wb")
    pickle.dump(similarity_score_dict, f)
    f.close()



