# -*- coding: utf-8 -*-
"""
Created on Wed Sep  1 13:32:11 2021

@author: Anne Saulin
"""

#######################################################################
## !!! make sure old model files and traces are not in the same folder 
## as they will be overwritten in the current version of the script

# test different p_outlier values

import pandas as pd
import hddm
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import numpy as np
import pickle

####### read in data 
# look at data
data_allemp = pd.read_csv("ddm_empathy_lab_scanner.csv", sep=",")
#data = df.loc[:, ['ID', 'rt', 'condition', 'resp_code', 'other_poss_loss']]

#data = data.rename(columns={"ID": "subj_idx", "resp_code": "response"})
data_allemp.rt = data_allemp.rt / 1000
print(data_allemp.head(n=6))

# work around enabling proper saving of the models (may not alyways be needed)
def my_save(self, fname):
    import pickle
    with open(fname, 'wb') as f:
        pickle.dump(self, f)

hddm.HDDM.my_save = my_save

## run models ###

# create model
model_base_allemp_p05= hddm.HDDM(data_allemp, bias=True, include=['z', 'st', 'sz', 'sv'], p_outlier=0.05)
# find a good starting point which helps with the convergence.
model_base_allemp_p05.find_starting_values()
# start drawing 5000 samples and discarding 2000 as burn-in
model_base_allemp_p05.sample(100, burn=50, dbname='base_traces_allemp_p05.db', db='pickle')
# model.save('bias_include_all/mymodel')
hddm.HDDM.my_save(model_base_allemp_p05,'model_base_allemp_p05')
m_reg_base_allemp_p05 = hddm.load('model_base_allemp_p05')


# create model
model_simple_allemp_p05= hddm.models.HDDMRegressor(data_allemp, 'v ~ other_poss_loss',
                                  bias=True, include=['z', 'st', 'sz', 'sv'], p_outlier=0.05, 
                                  group_only_regressors = False , keep_regressor_trace = True)
# find a good starting point which helps with the convergence.
model_simple_allemp_p05.find_starting_values()
# start drawing 5000 samples and discarding 2000 as burn-in
model_simple_allemp_p05.sample(2000, burn=1000, dbname='simple_traces_allemp_p05.db', db='pickle')
# model.save('bias_include_all/mymodel')
hddm.HDDM.my_save(model_simple_allemp_p05,'model_reg_simple_allemp_p05')
m_reg_simple_allemp_p05 = hddm.load('model_reg_simple_allemp_p05')



#create model
model_cond_v_allemp_p05= hddm.models.HDDMRegressor(data_allemp, 'v ~ other_poss_loss+condition', 
                                  bias=True, include=['z', 'st', 'sz', 'sv'], p_outlier=0.05, 
                                  group_only_regressors = False , 
                                  keep_regressor_trace = True)
# find a good starting point which helps with the convergence.
model_cond_v_allemp_p05.find_starting_values()
# start drawing 5000 samples and discarding 2000 as burn-in
model_cond_v_allemp_p05.sample(2000, burn=1000, dbname='cond_v_allemp_traces_p05.db', db='pickle')
# model.save('bias_include_all/mymodel')
hddm.HDDM.my_save(model_cond_v_allemp_p05,'model_reg_cond_v_allemp_p05')
m_reg_cond_v_allemp_p05 = hddm.load('model_reg_cond_v_allemp_p05')

#create model
model_cond_v_add_allemp_p05= hddm.models.HDDMRegressor(data_allemp, 'v ~ other_poss_loss+condition + C(blockno)', 
                                  bias=True, include=['z', 'st', 'sz', 'sv'], p_outlier=0.05, 
                                  group_only_regressors = False , 
                                  keep_regressor_trace = True)
# find a good starting point which helps with the convergence.
model_cond_v_add_allemp_p05.find_starting_values()
# start drawing 5000 samples and discarding 2000 as burn-in
model_cond_v_add_allemp_p05.sample(2000, burn=1000, dbname='cond_v_add_traces_allemp_p05.db', db='pickle')
# model.save('bias_include_all/mymodel')
hddm.HDDM.my_save(model_cond_v_add_allemp_p05,'model_reg_cond_v_add_allemp_p05')
m_reg_cond_v_add_allemp_p05 = hddm.load('model_reg_cond_v_add_allemp_p05')


#create model
model_cond_z_allemp_p05= hddm.models.HDDMRegressor(data_allemp, ['v ~ other_poss_loss',
                                              'z ~ condition'], 
                                  bias=True, include=['z', 'st', 'sz', 'sv'], p_outlier=0.05, 
                                  group_only_regressors = False , 
                                  keep_regressor_trace = True)
# find a good starting point which helps with the convergence.
model_cond_z_allemp_p05.find_starting_values()
# start drawing 5000 samples and discarding 2000 as burn-in
model_cond_z_allemp_p05.sample(2000, burn=1000, dbname='cond_z_traces_allemp_p05.db', db='pickle')
# model.save('bias_include_all/mymodel')
hddm.HDDM.my_save(model_cond_z_allemp_p05,'model_reg_cond_z_allemp_p05')
m_reg_cond_z_allemp_p05 = hddm.load('model_reg_cond_z_allemp_p05')

#create model
model_cond_z_add_allemp_p05= hddm.models.HDDMRegressor(data_allemp, ['v ~ other_poss_loss',
                                              'z ~ condition+C(blockno)'], 
                                  bias=True, include=['z', 'st', 'sz', 'sv'], p_outlier=0.05, 
                                  group_only_regressors = False , keep_regressor_trace = True)
# find a good starting point which helps with the convergence.
model_cond_z_add_allemp_p05.find_starting_values()
# start drawing 5000 samples and discarding 2000 as burn-in
model_cond_z_add_allemp_p05.sample(2000, burn=1000, dbname='cond_z_add_allemp_traces_p05.db', db='pickle')
# model.save('bias_include_all/mymodel')
hddm.HDDM.my_save(model_cond_z_add_allemp_p05,'model_reg_cond_z_add_allemp_p05')
m_reg_cond_z_add_allemp_p05 = hddm.load('model_reg_cond_z_add_allemp_p05')

#create model
model_cond_z_int_allemp_p05= hddm.models.HDDMRegressor(data_allemp, ['v ~ other_poss_loss',
                                              'z ~ condition*C(blockno)'], 
                                  bias=True, include=['z', 'st', 'sz', 'sv'], p_outlier=0.05, 
                                  group_only_regressors = False , keep_regressor_trace = True)
# find a good starting point which helps with the convergence.
model_cond_z_int_allemp_p05.find_starting_values()
# start drawing 5000 samples and discarding 2000 as burn-in
model_cond_z_int_allemp_p05.sample(2000, burn=1000, dbname='cond_z_int_allemp_traces_p05.db', db='pickle')
# model.save('bias_include_all/mymodel')
hddm.HDDM.my_save(model_cond_z_int_allemp_p05,'model_reg_cond_z_int_allemp_p05')
m_reg_cond_z_int_allemp_p05 = hddm.load('model_reg_cond_z_int_allemp_p05')

#####################################################################################################

######################################
# posterior predictive checks 
ppc_z_cond_int_allemp = hddm.utils.post_pred_gen(m_reg_cond_z_int_allemp_p05)
ppc_compare_allemp = hddm.utils.post_pred_stats(data, ppc_z_cond_int_allemp)
ppc_stats_z_cond_int_allemp = pd.DataFrame(ppc_compare_allemp)
ppc_stats_z_cond_int_allemp.to_csv('ppc_stats_model_z_cond_int_allemp.csv')

#### plot densities and hypothesis testing

####### plot posteriors of winning model
z_cond_block_int_allemp = m_reg_cond_z_int_allemp_p05.get_traces()


# effect of other possible gain
sns.distplot(z_cond_block_int_allemp['v_other_poss_loss'], hist=True, kde=True, rug =False,
             bins=int(180/5), color = 'darkblue', 
             hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth': 4},
             axlabel = 'other possible gain')

# effect of block number
t2greatert1 = list(z_cond_block_int_allemp['z_C(blockno)[T.2]']>0)
t2greatert1.count(True) # =949
sns.distplot(z_cond_block_int_allemp['z_C(blockno)[T.2]'], hist=True, kde=True, 
             bins=int(180/5), color = 'darkblue', 
             hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth': 4},
             axlabel = 'z block 2 > z block 1')

t3greatert1 = list(z_cond_block_int_allemp['z_C(blockno)[T.3]']>0)
t3greatert1.count(True) # =988
sns.distplot(z_cond_block_int_allemp['z_C(blockno)[T.3]'], hist=True, kde=True,
             bins=int(180/5), color = 'darkblue', 
             hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth': 4},
             axlabel = 'z block 3 > z block 1')

t2greatert3 = list(z_cond_block_int_allemp['z_C(blockno)[T.3]']-z_cond_block_int_allemp['z_C(blockno)[T.2]']>0)
t2greatert3.count(True) # 144
sns.distplot(z_cond_block_int_allemp['z_C(blockno)[T.2]']-z_cond_block_int_allemp['z_C(blockno)[T.3]'], hist=True, kde=True,
             bins=int(180/5), color = 'darkblue', 
             hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth': 4},
             axlabel = 'z block 2 > z block 3')


# effect of condition
treatgreatercon = list(z_cond_block_int_allemp['z_condition[T.treatment]']>0)
treatgreatercon.count(True) # =877
sns.distplot(z_cond_block_int_allemp['z_condition[T.treatment]'], hist=True, kde=True, 
             bins=int(180/5), color = 'darkblue', 
             hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth': 4},
             axlabel = 'z treatment > z control')


# interaction effects block:condition
sns.distplot(z_cond_block_int_allemp['z_condition[T.treatment]:C(blockno)[T.2]'], hist=True, kde=True, 
             bins=int(180/5), color = 'darkblue', 
             hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth': 4},
             axlabel = 'z interaction condition block2>block1')
sns.distplot(z_cond_block_int_allemp['z_condition[T.treatment]:C(blockno)[T.3]'], hist=True, kde=True, 
             bins=int(180/5), color = 'darkblue', 
             hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth': 4},
             axlabel = 'z interaction treatment: block3>block1')
sns.distplot(z_cond_block_int_allemp['z_condition[T.treatment]:C(blockno)[T.2]']-z_cond_block_int_allemp['z_condition[T.treatment]:C(blockno)[T.3]'], hist=True, kde=True, 
             bins=int(180/5), color = 'darkblue', 
             hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth': 4},
             axlabel = 'z interaction treatment block2>block3')

int_t2_t3 = list((z_cond_block_int_allemp['z_condition[T.treatment]:C(blockno)[T.3]']-z_cond_block_int_allemp['z_condition[T.treatment]:C(blockno)[T.2]'])<0)
int_t2_t3.count(True) # 868

int_t1_t3 = list((z_cond_block_int_allemp['z_condition[T.treatment]:C(blockno)[T.3]'])<0)
int_t1_t3.count(True) # 904

int_t1_t2 = list((z_cond_block_int_allemp['z_condition[T.treatment]:C(blockno)[T.2]'])>0)
int_t1_t2.count(True) # 411

# extract statistics
m_reg_cond_z_int_allemp_p05_stats = pd.DataFrame(m_reg_cond_z_int_allemp_p05.gen_stats())
# model_stats.to_csv
m_reg_cond_z_int_allemp_p05_stats.to_csv('m_reg_cond_z_int_allemp_p05_final_stats.csv')


######################################
ppc_z_cond_int_allemp = hddm.utils.post_pred_gen(m_reg_cond_z_int_allemp_p05)
ppc_compare_allemp = hddm.utils.post_pred_stats(data, ppc_z_cond_int_allemp)
ppc_stats_z_cond_int_allemp = pd.DataFrame(ppc_compare_allemp)
ppc_stats_z_cond_int_allemp.to_csv('ppc_stats_model_z_cond_int_allemp.csv')

