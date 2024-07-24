#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pickle
import pandas as pd


# In[ ]:


datadir = "/lapp_data/atlas/chardong/Venv/savedir/save_pkl/"

totald = pd.read_pickle(datadir+"Py8_yj_jj_mc16ade_pd122_train_w_skim_noFF.pkl")

# Vérifier si les données sont déjà dans un DataFrame
if isinstance(data, pd.DataFrame):
    df = data
else:
    # Si les données sont sous forme de dictionnaire de listes ou autre structure compatible
    df = pd.DataFrame(data)

# Sélectionner un échantillon de 30 % des données de manière aléatoire
df_sampled = df.sample(frac=0.3, random_state=42)

# Sauvegarder l'échantillon dans un nouveau fichier pickle
with open('/lapp_data/atlas/chardong/Venv/savedir/save_pkl/Py8_yj_jj_mc16ade_pd122_train_w_skim_30_noFF_sampled.pkl', 'wb') as f:
    pickle.dump(df_sampled, f)

print("Un échantillon de 30 % des données a été sauvegardé.")


# In[ ]:




