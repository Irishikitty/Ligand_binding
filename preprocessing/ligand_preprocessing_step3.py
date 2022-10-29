import requests
from bs4 import BeautifulSoup
import re
import pickle
import os
import numpy as np
import copy
from preprocessing_utils import *

print('Here',os.getcwd())
file = pickle.load(open(os.getcwd() + "/all_ligand_atomtype_chain_axes.pickle", "rb"))

def f_exact_ligand_len(ligand_name):
    URL = 'https://www.rcsb.org/ligand/' + ligand_name
    page = requests.get(URL)
    soup = BeautifulSoup(page.content, "html.parser")
    table_tags = soup.find_all('table')

    # true length
    tag = table_tags[3]
    tag = tag.find_all('tr')[2]
    length = int(tag.find('td').text)

    # H length
    tag = table_tags[2]
    formula = tag.find_all('tr', {'id': 'chemicalFormula'})[0].find('td').getText()
    length_H = re.search('H[\d]+', formula)

    if length_H:
        return length - int(length_H.group(0)[1:])
    else:
        return length

def get_single_ligand_length(file):

    ligands_len = dict()
    for i in file.keys():
        try:
            sub_ligands = i.split(" ")
            if len(sub_ligands) == 1 and i not in ligands_len.keys():
                ligands_len[i] = f_exact_ligand_len(i)
            else:
                for sub_ligand in sub_ligands:
                    if sub_ligand not in ligands_len.keys():
                        ligands_len[sub_ligand] = f_exact_ligand_len(sub_ligand)
        except:
            print(i)
    return ligands_len


def get_ligand_compound_length(file, ligand_len):

    for ligand in file.keys():
        molecules = ligand.split(' ')
        if len(molecules)>1:
            assert ligand_len.get(ligand, None) == None
            for j in molecules:
                if ligand_len.get(j, None) is None:
                # assert ligand_len.get(j, None) == None
                    ligand_len[j] = f_exact_ligand_len(j)
            temp  = [ligand_len[j] for j in molecules]
            ligand_len[ligand] = np.sum(temp)
            print("get ligand compound length: ", ligand)
            # print(len(file) - len(ligand_len))

    a_file = open("ligand_len.pickle", "wb")
    pickle.dump(ligand_len, a_file)
    a_file.close()

    return ligand_len


if __name__=='__main__':

    pdb_files_dir = os.getcwd() + '/pdb_files_all_new/'
    file = pickle.load(open("all_ligand_pdbid_chain_atomtype_axes_pos.pickle", "rb"))
    print(len(file.keys()))

    length_dict = get_single_ligand_length(file)
    length_dict_final = get_ligand_compound_length(file, length_dict)
    print(length_dict_final)
    print(len(length_dict_final.keys()))

    ligand_length_dict = pickle.load(open("ligand_len.pickle", "rb"))
    print(len(ligand_length_dict.keys()))

    new_file = {}
    for ligand, pdbinfo in file.items():
        true_ligand_length = ligand_length_dict[ligand]
        new_pdbinfo_list = [pdb for pdb in pdbinfo if len(pdb[2]) == true_ligand_length]
        if new_pdbinfo_list:
            new_file[ligand] = new_pdbinfo_list

    print(len(list(new_file.keys())))


    f = open("all_ligand_pdbid_chain_atomtype_axes_pos_distill.pickle","wb")
    pickle.dump(new_file,f)
    f.close()


    import pickle
    new_file = pickle.load(open("all_ligand_pdbid_chain_atomtype_axes_pos_distill.pickle", "rb"))   # {ligand: [[pdbid, chainID, ligand_atom, axes, pos]]}
    counter = 0
    pdb_chainid_list = []
    for key, value in new_file.items():
        for ele in value:
            if (ele[0], ele[1]) not in pdb_chainid_list:
                pdb_chainid_list.append((ele[0], ele[1]))
                counter += 1
    print(counter)

    f = open("pdb_chainid_list.pickle","wb")
    pickle.dump(pdb_chainid_list,f)
    f.close()








    # length = f_exact_ligand_len('9qd')
    # print(f_exact_ligand_len('ZZZ'))
    # print(f_exact_ligand_len('Cu'))
    # url = URL = 'https://www.rcsb.org/ligand/' + '9qd'
    # r = requests.get(url)
    # df_list = pd.read_html(r.text)  # this parses all the tables in webpages to a list
    # df = df_list[0]
    # df.head()

# 408 51 ACO 2Q4V
# 176 23 AMP 2Q4H
# 176 23 AMP 2Q4H
# 275 12 BGC 4TZ3
# 300 12 BGC 4TYV
# 167 1 CA 4P3Q
# 104 13 CIT 2Q4G
# 192 48 COA 2Q4Y
# 848 53 FAD 2Q4W
# 496 31 FMN 2Q3R
# 496 31 FMN 2Q3O
# 496 31 FMN 2Q3O
# 8000 32 FOL 4P3R
# 5344 32 FOL 4P3Q
# 8000 32 FOL 4PTH
# 4000 32 FOL 4PTJ
# 22 12 HC4 1OT6
# 22 12 HC4 1OT9
# 80 10 HMH 2Q4X
# 80 10 HMH 2Q4X
# 112 7 MLA 2Q3M
# 250 1 MN 4PTH
# 250 1 MN 4PTH
# 125 1 MN 4PTJ
# 125 1 MN 4PTJ
# 12000 48 NAP 4P3R
# 8016 48 NAP 4P3Q
# 12000 48 NAP 4PTH
# 6000 48 NAP 4PTJ
# 768 48 NAP 2Q4B
# 768 48 NAP 2Q4B
# 384 48 NAP 2Q46
# 384 48 NAP 2Q46
# 49 18 PLM 6S2M
#
# 72 36 UPG 2ICY
#
# 8 1 ZN 2Q4H
# 8 1 ZN 2Q4H
# 8 1 ZN 2Q4H
# 8 1 ZN 2Q4H