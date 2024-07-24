#!/usr/bin/env python
# coding: utf-8

# # Photon ID Run 2 BDT training sample skimming

# In[3]:


import uproot
import numpy as np
import pandas as pd
import pickle


# In[4]:


pd.__version__


# In[5]:


datadir = "/lapp_data/atlas/perf-egamma/InclusivePhotons/fullRun2/FinalNtuples/"


# In[ ]:


df_sig = pd.read_pickle(datadir+"Py8_yj_mc16ade_pd122_train_w.pkl")
df_bkg = pd.read_pickle(datadir+"Py8_jj_mc16ade_pd122_train_w.pkl")


# In[5]:


columns = list(df_sig.columns)
columns


# In[19]:


# variables to keep

shower_shape_var = [ 'y_noFF_Reta',
 'y_noFF_Rphi',
 'y_noFF_weta2',
 'y_noFF_fracs1',
 'y_noFF_weta1',
 'y_noFF_wtots1',
 'y_noFF_Rhad',
 'y_noFF_Rhad1',
 'y_noFF_Eratio',
 'y_noFF_deltae']

angular_dist_min = ['y_jmin_dr']

isEM_var = [ 'y_IsTight', 'y_IsLoose' ]

conv_var = [ 'y_convRadius', 'y_convType']

kinem_var = ['y_pt', 'y_eta', 'y_phi', 'evt_mu']

truth_var = ['y_truth_pt', 'y_truth_eta' ] #, 'y_truth_type', 'y_truth_pdgId', 'y_truth_mother_pdgId' ]


# In[20]:


# add 'truth_label' variable to df
df_sig["truth_label"]=1.
df_bkg["truth_label"]=0.

# uniform weight column
df_sig["weight"]=df_sig['mcTotWeight']
df_bkg["weight"]=df_bkg["totWeight"]

# Not needed, already selected in previous step
# df_sig = df_sig.query('y_truth_type == 14')


# In[25]:


keep_var = shower_shape_var+angular_dist_min+conv_var+kinem_var+isEM_var+truth_var+['weight',"truth_label"]
df_sig_skim = df_sig[keep_var]
df_bkg_skim = df_bkg[keep_var]


# In[26]:


df_sig_skim.head(5)


# In[27]:


# merge signal and background samples
totald = pd.concat([df_sig_skim, df_bkg_skim], axis=0)


# In[28]:


# save skimmed dataframe
totald.to_pickle(datadir+"Py8_yj_jj_mc16ade_pd122_train_w_skim_noFF.pkl")


# In[ ]:




