#!/usr/bin/env python
# coding: utf-8

# # Photon ID Run 2 BDT classification noFF + FF

# In[1]:

import numpy as np
import pandas as pd
import pickle

# In[2]:


datadir = "/lapp_data/atlas/chardong/Venv/savedir/save_pkl/"
#savedir = "/home/chardong/y_identification/Venv/save_plots/Py8_yj_jj_train_skim30/"
# Chemin pour enregistrer les fichiers pickle
save_path = '/lapp_data/atlas/chardong/Venv/savedir/save_pkl/'
#datadir = "/eos/user/m/mdelmast/Data/EGamma/PhotonID/Run2/"
#savedirmodel = "/home/chardong/y_identification/Venv/BDT_model/"


# In[3]:


#totald = pd.read_pickle(datadir+"Py8_yj_jj_mc16ade_pd122_train_w.pkl")
#totald = pd.read_pickle(datadir+"Py8_yj_jj_mc16ade_pd122_train_w_skim.pkl")
totald = pd.read_pickle(datadir+"Py8_yj_jj_mc16ade_pd122_train_w_skim_30_noFF.pkl")
#totald_noFF = pd.read_pickle(datadir+"RAW_data/Py8_yj_jj_mc16ade_pd122_train_w_skim.pkl")


# In[4]:


# Définir la fonction de calcul des fudge factors
def FF(X, X_noFF):
    FFplus = X + (X_noFF - X) / 2
    FFminus = X - (X_noFF - X) / 2
    return FFplus, FFminus


# Liste des colonnes shower shape
shower_shape_columns = ['y_Reta', 'y_Rphi', 'y_weta2', 'y_fracs1', 'y_weta1', 'y_wtots1', 'y_Rhad', 'y_Rhad1', 'y_Eratio', 'y_deltae']

# Appliquer la fonction FF à chaque paire de colonnes
for col in shower_shape_columns:
    noFF_col = 'y_noFF_' + col.split('_')[1]
    if noFF_col in totald.columns:
        totald[f'{col}_FFplus'], totald[f'{col}_FFminus'] = FF(totald[col], totald[noFF_col])

# Afficher les 5 premières lignes du DataFrame
#print(totald.head(5))


# In[ ]:


# Créer et sauvegarder le DataFrame avec noFF
for col in shower_shape_columns:
    noFF_col = 'y_noFF_' + col.split('_')[1]
    totald[col] = totald[noFF_col]
    totald.drop(columns=[noFF_col], inplace=True)

with open(save_path + 'totald_noFF.pkl', 'wb') as f:
    pickle.dump(totald, f)
print("DataFrame avec noFF sauvegardé.")


# In[ ]:


# Recharger le DataFrame initial pour créer et sauvegarder le DataFrame avec FFminus
totald = pd.read_pickle(datadir+"Py8_yj_jj_mc16ade_pd122_train_w_skim_30_noFF.pkl")
for col in shower_shape_columns:
    noFF_col = 'y_noFF_' + col.split('_')[1]
    if noFF_col in totald.columns:
        totald[f'{col}_FFplus'], totald[f'{col}_FFminus'] = FF(totald[col], totald[noFF_col])

for col in shower_shape_columns:
    totald[col] = totald[f'{col}_FFminus']
    totald.drop(columns=[f'{col}_FFminus'], inplace=True)
    totald.drop(columns=[f'{col}_FFplus'], inplace=True)  # Supprimer les colonnes FFplus

with open(save_path + 'totald_FFminus.pkl', 'wb') as f:
    pickle.dump(totald, f)
print("DataFrame avec FFminus sauvegardé.")

# Recharger le DataFrame initial pour créer et sauvegarder le DataFrame avec FFplus
totald = pd.read_pickle(datadir+"Py8_yj_jj_mc16ade_pd122_train_w_skim_30_noFF.pkl")
for col in shower_shape_columns:
    noFF_col = 'y_noFF_' + col.split('_')[1]
    if noFF_col in totald.columns:
        totald[f'{col}_FFplus'], totald[f'{col}_FFminus'] = FF(totald[col], totald[noFF_col])

for col in shower_shape_columns:
    totald[col] = totald[f'{col}_FFplus']
    totald.drop(columns=[f'{col}_FFplus'], inplace=True)
    totald.drop(columns=[f'{col}_FFminus'], inplace=True)  # Supprimer les colonnes FFminus

with open(save_path + 'totald_FFplus.pkl', 'wb') as f:
    pickle.dump(totald, f)
print("DataFrame avec FFplus sauvegardé.")


