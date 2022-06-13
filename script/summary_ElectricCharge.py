#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 12 21:27:54 2022

@author: menghanlin
"""
import os
os.listdir('./')

import requests
from bs4 import BeautifulSoup
import re
import pickle
import time
import numpy as np

print('Here',os.getcwd())
file = pickle.load(open("./test_ligand_env_coords.pickle", "rb"))

def f_exact_ligand_charge(ligand_name):
    URL = 'https://www.rcsb.org/ligand/' + ligand_name
    page = requests.get(URL)
    soup = BeautifulSoup(page.content, "html.parser")

    # charge
    tag = soup.find_all('table')[1]
    tag = tag.find_all('tr')[1]
    charge = int(tag.find('td').text)

    
    return charge



# ligand_len = dict()
# counter = 0
# t1 = time.time()
# for i in set(file.keys()):
#     try:
#         ligand_len[i] = f_exact_ligand_len(i)
#         counter += 1
#         if (counter % 1000)==0:
#             print(time.time() - t1)
#     except:
#         print(i)



# Make up ligands with more molecules
df_positive_ligand = []
df_negative_ligand = []
df_neuter_ligand = []

for i, v in enumerate(file):
    t1 = time.time()
    
    ligand_name = v[1]
    ligand_charge = f_exact_ligand_charge(ligand_name)
    molecules = ligand_name.split(' ')
    # if len(molecules)>1:
    #     accu = 0
    #     for j in molecules:
    #         accu += f_exact_ligand_charge(j)
    
        
    print(time.time()-t1)
    print(f'ligand name: {ligand_name}   ligand charge: {ligand_charge}')
    if ligand_charge == 0:
        df_neuter_ligand.append(v)
    elif ligand_charge > 0:
        df_positive_ligand.append(v)
    elif ligand_charge < 0:
        df_negative_ligand.append(v)
    
        
df_positive_ligand = open("df_positive_ligand.pkl", "wb")
pickle.dump(ligand_len, df_positive_ligand)
df_positive_ligand.close()

df_negative_ligand = open("df_negative_ligand.pkl", "wb")
pickle.dump(ligand_len, df_negative_ligand)
df_negative_ligand.close()

df_neuter_ligand = open("df_neuter_ligand.pkl", "wb")
pickle.dump(ligand_len, df_neuter_ligand)
df_neuter_ligand.close()













