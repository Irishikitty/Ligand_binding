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
from collections import Counter
import json
import numpy as np

file = pickle.load(open("./train_ligand_env_coords.pickle", "rb"))

temp = 'train' # test or train
file_train = file
# file_test = file

def f_exact_ligand_charge(ligand_name):
    
    URL = 'https://www.rcsb.org/ligand/' + ligand_name
    page = requests.get(URL)
    soup = BeautifulSoup(page.content, "html.parser")

    # charge
    tag = soup.find_all('table')[1]
    tag = tag.find_all('tr')[1]
    charge = int(tag.find('td').text)
    
    return charge

lst = [i[1] for i in file]
lst = Counter(lst).keys()
df_charge_dict = {}

for i, v in enumerate(lst):
    t1 = time.time()
    
    ligand_name = v
    ligand_charge = f_exact_ligand_charge(ligand_name)
    molecules = ligand_name.split(' ')
    
    # save charge info
    df_charge_dict[ligand_name] = ligand_charge
    print(f'Time used: {np.round(time.time()-t1,1)}     ligand name: {ligand_name}   ligand charge: {ligand_charge}')

# create json object from dictionary
json = json.dumps(df_charge_dict)
f = open("df_charge_dict.json","w")
f.write(json)
f.close()

# Make up ligands with more molecules
df_positive_ligand = []
df_negative_ligand = []
df_neuter_ligand = []

for i, v in enumerate(file):
    ligand_name = v[1]
    ligand_charge = df_charge_dict[ligand_name]
    
    
    
    if ligand_charge == 0:
        df_neuter_ligand.append(v)
    elif ligand_charge > 0:
        df_positive_ligand.append(v)
    elif ligand_charge < 0:
        df_negative_ligand.append(v)
    
positive_ligand = open("./charge/df_"+temp+"_positive_ligand.pkl", "wb")
pickle.dump(df_positive_ligand, positive_ligand)
positive_ligand.close()

negative_ligand = open("./charge/df_"+temp+"negative_ligand.pkl", "wb")
pickle.dump(df_negative_ligand, negative_ligand)
negative_ligand.close()

neuter_ligand = open("./charge/df"+temp+"neuter_ligand.pkl", "wb")
pickle.dump(df_neuter_ligand, neuter_ligand)
neuter_ligand.close()













