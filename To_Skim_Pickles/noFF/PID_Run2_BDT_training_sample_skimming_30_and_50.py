#!/usr/bin/env python
# coding: utf-8

# # Photon ID Run 2 BDT training sample skimming

# In[1]:


import uproot
import numpy as np
import pandas as pd
import pickle


# In[2]:


pd.__version__


# In[3]:


datadir = "/lapp_data/atlas/perf-egamma/InclusivePhotons/fullRun2/FinalNtuples/"
datasave = "/lapp_data/atlas/chardong/Venv/savedir/save_pkl/"


# In[4]:


df_sig = pd.read_pickle(datadir+"Py8_yj_mc16ade_pd122_train_w.pkl")
df_bkg = pd.read_pickle(datadir+"Py8_jj_mc16ade_pd122_train_w.pkl")


# In[5]:


columns = list(df_sig.columns)
columns

df_sig['y_Reta_FFplus'] = df_sig['y_Reta']+(df_sig['y_noFF_Reta']-df_sig['y_Reta'])/2
df_sig['y_Reta_FFminus'] = df_sig['y_Reta']-(df_sig['y_noFF_Reta']-df_sig['y_Reta'])/2
df_sig.head(5)
# In[6]:


# variables to keep

shower_shape_var = ['y_Reta',
                    'y_Rphi',
                    'y_weta2',
                    'y_fracs1',
                    'y_weta1',
                    'y_wtots1',
                    'y_Rhad',
                    'y_Rhad1',
                    'y_Eratio', 
                    'y_deltae',
                   'y_noFF_Reta',
                   'y_noFF_Rphi',
                   'y_noFF_weta2',
                   'y_noFF_fracs1',
                   'y_noFF_weta1',
                   'y_noFF_wtots1',
                   'y_noFF_Rhad',
                   'y_noFF_Rhad1',
                   'y_noFF_Eratio',
                   'y_noFF_deltae']

isEM_var = [ 'y_IsTight', 'y_IsLoose' ]

conv_var = [ 'y_convRadius', 'y_convType']

kinem_var = ['y_pt', 'y_eta', 'y_phi', 'evt_mu', 'y_jmin_dr']

truth_var = ['y_truth_pt', 'y_truth_eta' ] #, 'y_truth_type', 'y_truth_pdgId', 'y_truth_mother_pdgId' ]


# In[7]:


# add 'truth_label' variable to df
df_sig["truth_label"]=1.
df_bkg["truth_label"]=0.

# uniform weight column
df_sig["weight"]=df_sig['mcTotWeight']
df_bkg["weight"]=df_bkg["totWeight"]

# Not needed, already selected in previous step
# df_sig = df_sig.query('y_truth_type == 14')


# In[8]:


keep_var = shower_shape_var+conv_var+kinem_var+isEM_var+truth_var+['weight',"truth_label"]
df_sig_skim = df_sig[keep_var]
df_bkg_skim = df_bkg[keep_var]


# In[9]:


df_sig_skim.head(5)


# In[10]:


# further reduce dataset size by downsampling (e.g. 50% of the data)
df_sig_skim_50 = df_sig_skim.sample(frac=0.5) 
df_bkg_skim_50 = df_bkg_skim.sample(frac=0.5)

df_sig_skim_30 = df_sig_skim.sample(frac=0.3) 
df_bkg_skim_30 = df_bkg_skim.sample(frac=0.3)


# In[11]:


# merge signal and background samples
totald = pd.concat([df_sig_skim, df_bkg_skim], axis=0)
# save skimmed dataframe
totald.to_pickle(datasave+"Py8_yj_jj_mc16ade_pd122_train_w_skim_noFF.pkl")


# In[12]:


totald = pd.concat([df_sig_skim_50, df_bkg_skim_50], axis=0)
totald.to_pickle(datasave+"Py8_yj_jj_mc16ade_pd122_train_w_skim_50_noFF.pkl")


# In[13]:


totald = pd.concat([df_sig_skim_30, df_bkg_skim_30], axis=0)
totald.to_pickle(datasave+"Py8_yj_jj_mc16ade_pd122_train_w_skim_30_noFF.pkl")


# In[ ]:




