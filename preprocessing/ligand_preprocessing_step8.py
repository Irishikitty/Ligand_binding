import pickle
from tqdm import tqdm
from collections import deque
import numba as nb
import numpy as np
import scipy.spatial.distance as ssd
import scipy.cluster.hierarchy as sch
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


def Merge(dict1, dict2):
    return(dict2.update(dict1))

@nb.njit()
def upp2sym(A):
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            A[j, i] = A[i, j]
    return A


def plot_corr(df, size=10):
    '''Plot a graphical correlation matrix for a dataframe.

    Input:
        df: pandas DataFrame
        size: vertical and horizontal size of the plot'''


    import matplotlib.pyplot as plt

    # Compute the correlation matrix for the received dataframe
    corr = df.corr()

    # Plot the correlation matrix
    fig, ax = plt.subplots(figsize=(size, size))
    cax = ax.matshow(corr, cmap='RdYlGn')
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)

    # Add the colorbar legend
    cbar = fig.colorbar(cax, ticks=[-1, 0, 1], aspect=40, shrink=.8)


def test(ind):
    for index in list(set(ind)):
        indexes_list = np.where(ind == index)[0].tolist()
        if len(indexes_list) > 1:
            for i in range(len(indexes_list)):
                for j in range(i+1, len(indexes_list)):
                    print(correlation_matrix_sym[indexes_list[i], indexes_list[j]])
                    # if correlation_matrix_sym[indexes_list[i], indexes_list[j]] >= 0.7:
                    #     print('wrong clusters!!!')

def toSeqIndex(ind_list):
    return [f'seq_{i}' for i in ind_list]

def SeqIndex_2_Seq(seqIndexList, index2seq):
    return [index2seq[seq_index] for seq_index in seqIndexList]

def seqlist_2_ligandInfo(seqlist, sequence_2_ligandInfo):
    ligand_required_info = []
    for seq in seqlist:
        ligand_required_info.extend(sequence_2_ligandInfo[seq])
    return ligand_required_info


if __name__ == "__main__":


    '''    
    filename_list = ['1-100_201-250_400-500', '300-349','101-200_251-299', '350-399', '501-516']
    similarity_score_dict_all = {}
    key_name_list = []
    for file in filename_list:
        similarity_score_dict = pickle.load(open(f'similarity_score_dict_{file}.pickle', 'rb'))
        print(len(similarity_score_dict.keys()))
        key_name_list += list(similarity_score_dict.keys())
        Merge(similarity_score_dict, similarity_score_dict_all)

    f = open("similarity_score_dict_all.pickle","wb")
    pickle.dump(similarity_score_dict_all, f)
    f.close()

    print('Done!')
    '''

    # filename = 'similarity_score_dict_all.pickle'
    # similarity_score_dict_all = pickle.load(open(filename, 'rb'))                                                       # {'seq_mix_1': {(seq_1, seq_2): 25.0, (seq_1, seq_3): 30.0}}
    #
    # index_2_seq = pickle.load(open('seq_info/index_2_seq.pickle', 'rb'))
    #
    # sequence_2_ligandInfo = pickle.load(open("sequence_2_ligandInfo.pickle", "rb"))
    #
    # correlation_matrix = np.zeros((17590, 17590))
    # one_matrix = np.ones((17590, 17590)) - np.eye(17590)
    # seqA_index_list = deque()
    # counter = 0
    # for _, seq_scores in tqdm(similarity_score_dict_all.items()):
    #     for seq_indexes, score in seq_scores.items():
    #         seqA_index, seqB_index = seq_indexes
    #         i, j = seqA_index.strip('seq_'), seqB_index.strip('seq_')
    #         correlation_matrix[int(i),int(j)] = score
    #
    # correlation_matrix_sym = upp2sym(correlation_matrix)
    # correlation_matrix_sym = correlation_matrix_sym / 100
    # distance_matrix = one_matrix - correlation_matrix_sym
    #
    # condensed_correlation_matrix_sym = ssd.squareform(correlation_matrix_sym)
    # condensed_distance_matrix = ssd.squareform(distance_matrix)


    # condensed_distance_matrix = pickle.load(open("condensed_distance_matrix.pickle","rb"))
    # L = sch.linkage(condensed_distance_matrix, method='complete')
    # import matplotlib.pyplot as plt
    # plt.figure(figsize=(40,10), dpi=150)
    # sch.dendrogram(L, color_threshold=0.3, above_threshold_color='lightgray')
    # plt.axhline(c='b', linestyle='--', y=0.3)
    # # plt.show()
    # plt.savefig('dendrogram.jpg')
    # plt.close()
    #
    #
    #
    #
    #
    # clusterIDs = sch.fcluster(L, 0.3, 'distance')
    #
    # unique_clusterIDs = list(set(clusterIDs))
    #
    # '''
    #     unique_clusterIDs: cluster ID for each seq_index
    # '''
    #
    # train_clusterIDs, test_clusterIDs = train_test_split(unique_clusterIDs, test_size = 0.2, random_state = 42)
    # train_clusterIDs, validation_clusterIDs = train_test_split(train_clusterIDs, test_size = 0.125, random_state = 42)
    #
    # print(len(train_clusterIDs))
    #
    # dataset = {'train':[], 'validation': [], 'test': []}
    # for count, target_clusterIDs in enumerate([train_clusterIDs, validation_clusterIDs, test_clusterIDs]):
    #     target_index_list = []
    #     for clusterID in target_clusterIDs:
    #         index = np.where(clusterIDs == clusterID)[0].tolist()                   # collect all seq_indexes associated with the current cluster ID
    #         target_index_list += index
    #
    #     target_seq_index_list = toSeqIndex(target_index_list)               # from cluster ID to seq_index
    #     seq_list = SeqIndex_2_Seq(target_seq_index_list, index_2_seq)
    #     ligand_required_data = seqlist_2_ligandInfo(seq_list, sequence_2_ligandInfo)
    #
    #     if count == 0:
    #         dataset['train'].extend(ligand_required_data)
    #     if count == 1:
    #         dataset['validation'].extend(ligand_required_data)
    #     if count ==2:
    #         dataset['test'].extend(ligand_required_data)
    #
    # f = open("dataset.pickle","wb")
    # pickle.dump(dataset,f)
    # f.close()
    #
    # print('Done!')



    dataset = pickle.load(open("dataset.pickle","rb"))
    print(dataset)
    length_list = {'train': [], 'validation': [], 'test': []}
    for key, value in dataset.items():
        for data in value:
            length_list[key].append(len(data[2]))

    print(length_list)

    for key, length in length_list.items():
        plt.figure(figsize = (10, 8), dpi = 300)
        plt.hist(length, bins = 100)
        plt.xlim(0, 100)
        plt.xticks(fontsize = 24)
        plt.yticks(fontsize = 24)
        plt.ylabel('ligand length', fontsize = 24)
        plt.axvline(50, linestyle='--', c = 'red')
        plt.title(f'{key} dataset', fontsize = 24)
        plt.savefig(f'{key}_length.jpg', bbox_inches='tight')