"""
Created on Tue Feb 22 14:07:31 2022

@author: menghanlin
Extract ligand information from PDB, count atom length and  # H
Return a dictionary key = ligand name, value = # atoms - # H

"""
import requests
from bs4 import BeautifulSoup
import re
import pickle
import os
import time
import numpy as np

print('Here',os.getcwd())
file = pickle.load(open("./all_ligand_atom_type.pkl", "rb"))

def f_exact_ligand_len(ligand_name):
    URL = 'https://www.rcsb.org/ligand/' + ligand_name
    page = requests.get(URL)
    soup = BeautifulSoup(page.content, "html.parser")

    # true length
    tag = soup.find_all('table')[1]
    tag = tag.find_all('tr')[2]
    length = int(tag.find('td').text)

    # H length
    tag = soup.find_all('table')[0]
    tag = tag.find_all('tr')[3]
    atom_type = tag.find('td').getText()
    m = re.search('H[\d]+', atom_type)
    
    tag = soup.find_all('table')[0]
    tag = tag.find_all('tr')[4]
    atom_type = tag.find('td').getText()
    m1 = re.search('H[\d]+', atom_type)
    
    if m is not None:
        return length - int(m.group(0)[1:])
    elif m1 is not None:
        return length - int(m1.group(0)[1:])
    elif (m is None) and (m1 is None):
        return length















