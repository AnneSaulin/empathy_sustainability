# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 11:34:40 2022

@author: annes
"""

#######################################################################
## !!! make sure old model files and traces are not in the same folder 
## as they will be overwritten in the current version of the script

# test different p_outlier values

import pandas as pd
import hddm   # version 0.8.0
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import numpy as np
import pickle


# look at data
data_recip = pd.read_csv("ddm_reciprocity.csv", sep=",")

data_recip.rt = data_recip.rt / 1000
print(data_recip.head(n=6))


# work around enabling proper saving of the models (may not alyways be needed)
def my_save(self, fname):
    import pickle
    with open(fname, 'wb') as f:
        pickle.dump(self, f)

hddm.HDDM.my_save = my_save

# create model
model_recip_base_p05= hddm.HDDM(data_recip, bias=True, include=['z', 'st', 'sz', 'sv'], p_outlier=0.05)
# find a good starting point which helps with the convergence.
model_recip_base_p05.find_starting_values()
# start drawing 5000 samples and discarding 2000 as burn-in
model_recip_base_p05.sample(2000, burn=1000, dbname='base_traces_recip_p05.db', db='pickle')
# model.save('bias_include_all/mymodel')
hddm.HDDM.my_save(model_recip_base_p05,'model_recip_base_p05')
m_recip_reg_base_p05 = hddm.load('model_recip_base_p05')

#m_recip_reg_base_p05.plot_posteriors()

# create model
model_recip_simple_p05= hddm.models.HDDMRegressor(data_recip, 'v ~ other_poss_loss',
                                  bias=True, include=['z', 'st', 'sz', 'sv'], p_outlier=0.05, 
                                  group_only_regressors = False , keep_regressor_trace = True)
# find a good starting point which helps with the convergence.
model_recip_simple_p05.find_starting_values()
# start drawing 5000 samples and discarding 2000 as burn-in
model_recip_simple_p05.sample(2000, burn=1000, dbname='simple_traces_recip_p05.db', db='pickle')
# model.save('bias_include_all/mymodel')
hddm.HDDM.my_save(model_recip_simple_p05,'model_recip_reg_simple_p05')
m_recip_reg_simple_p05 = hddm.load('model_recip_reg_simple_p05')

#m_recip_reg_simple_p05.plot_posteriors()



#create model
model_recip_cond_v_p05= hddm.models.HDDMRegressor(data_recip, 'v ~ other_poss_loss+condition', 
                                  bias=True, include=['z', 'st', 'sz', 'sv'], p_outlier=0.05, 
                                  group_only_regressors = False , keep_regressor_trace = True)
# find a good starting point which helps with the convergence.
model_recip_cond_v_p05.find_starting_values()
# start drawing 5000 samples and discarding 2000 as burn-in
model_recip_cond_v_p05.sample(2000, burn=1000, dbname='cond_v_traces_recip_p05.db', db='pickle')
# model.save('bias_include_all/mymodel')
hddm.HDDM.my_save(model_recip_cond_v_p05,'model_recip_reg_cond_v_p05')
m_recip_reg_cond_v_p05 = hddm.load('model_recip_reg_cond_v_p05')

#create model
model_recip_cond_v_add_p05= hddm.models.HDDMRegressor(data_recip, 'v ~ other_poss_loss+condition+C(blockno)', 
                                  bias=True, include=['z', 'st', 'sz', 'sv'], p_outlier=0.05, 
                                  group_only_regressors = False , keep_regressor_trace = True)
# find a good starting point which helps with the convergence.
model_recip_cond_v_add_p05.find_starting_values()
# start drawing 5000 samples and discarding 2000 as burn-in
model_recip_cond_v_add_p05.sample(2000, burn=1000, dbname='cond_v_add_traces_recip_p05.db', db='pickle')
# model.save('bias_include_all/mymodel')
hddm.HDDM.my_save(model_recip_cond_v_add_p05,'model_recip_reg_cond_v_add_p05')
m_recip_reg_cond_v_add_p05 = hddm.load('model_recip_reg_cond_v_add_p05')


#create model
model_recip_cond_z_p05= hddm.models.HDDMRegressor(data_recip, ['v ~ other_poss_loss',
                                              'z ~ condition'],  
                                  bias=True, include=['z', 'st', 'sz', 'sv'], p_outlier=0.05,  
                                  group_only_regressors = False , keep_regressor_trace = True)
# find a good starting point which helps with the convergence.
model_recip_cond_z_p05.find_starting_values()
# start drawing 5000 samples and discarding 2000 as burn-in
model_recip_cond_z_p05.sample(2000, burn=1000, dbname='cond_z_traces_recip_p05.db', db='pickle')
# model.save('bias_include_all/mymodel')
hddm.HDDM.my_save(model_recip_cond_z_p05,'model_recip_reg_cond_z_p05')
m_recip_reg_cond_z_p05 = hddm.load('model_recip_reg_cond_z_p05')

#create model
model_recip_cond_z_add_p05= hddm.models.HDDMRegressor(data_recip, ['v ~ other_poss_loss',
                                              'z ~ condition+C(blockno)'], 
                                  bias=True, include=['z', 'st', 'sz', 'sv'], p_outlier=0.05, 
                                  group_only_regressors = False , keep_regressor_trace = True)
# find a good starting point which helps with the convergence.
model_recip_cond_z_add_p05.find_starting_values()
# start drawing 5000 samples and discarding 2000 as burn-in
model_recip_cond_z_add_p05.sample(2000, burn=1000, dbname='cond_z_add_traces_recip_p05.db', db='pickle')
# model.save('bias_include_all/mymodel')
hddm.HDDM.my_save(model_recip_cond_z_add_p05,'model_recip_reg_cond_z_add_p05')
m_recip_reg_cond_z_add_p05 = hddm.load('model_recip_reg_cond_z_add_p05')

#create model
model_recip_cond_z_int_p05= hddm.models.HDDMRegressor(data_recip, ['v ~ other_poss_loss',
                                              'z ~ condition*C(blockno)'], 
                                  bias=True, include=['z', 'st', 'sz', 'sv'], p_outlier=0.05, 
                                  group_only_regressors = False , keep_regressor_trace = True)
# find a good starting point which helps with the convergence.
model_recip_cond_z_int_p05.find_starting_values()
# start drawing 5000 samples and discarding 2000 as burn-in
model_recip_cond_z_int_p05.sample(2000, burn=1000, dbname='cond_z_int_traces_recip_p05.db', db='pickle')
# model.save('bias_include_all/mymodel')
hddm.HDDM.my_save(model_recip_cond_z_int_p05,'model_recip_reg_cond_z_int_p05')
m_recip_reg_cond_z_int_p05 = hddm.load('model_recip_reg_cond_z_int_p05')



#######################################################################################
# posterior predictive checks
ppc_z_cond_int_recip = hddm.utils.post_pred_gen(m_recip_reg_cond_z_int_p05)
ppc_compare_recip = hddm.utils.post_pred_stats(data_recip, ppc_z_cond_int_recip)
ppc_stats_z_cond_int_recip = pd.DataFrame(ppc_compare_recip)
ppc_stats_z_cond_int_recip.to_csv('ppc_stats_model_z_cond_int_recip.csv')


####### plot posteriors of winning model
z_cond_block_int_reciplab = m_recip_reg_cond_z_int_p05.get_traces()

# effect of other possible gain
sns.distplot(z_cond_block_int_reciplab['v_other_poss_loss'], hist=True, kde=True, 
             bins=int(180/5), color = 'darkblue', 
             hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth': 4},
             axlabel = 'other possible gain')

# effect of block number
sns.distplot(z_cond_block_int_reciplab['z_C(blockno)[T.2]'], hist=True, kde=True, 
             bins=int(180/5), color = 'darkblue', 
             hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth': 4})

sns.distplot(z_cond_block_int_reciplab['z_C(blockno)[T.3]'], hist=True, kde=True,
             bins=int(180/5), color = 'darkblue', 
             hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth': 4})

sns.distplot(z_cond_block_int_reciplab['z_C(blockno)[T.3]']-z_cond_block_int_reciplab['z_C(blockno)[T.2]'], hist=True, kde=True,
             bins=int(180/5), color = 'darkblue', 
             hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth': 4})


# effect of condition
sns.distplot(z_cond_block_int_reciplab['z_condition[T.treatment]'], hist=True, kde=True, 
             bins=int(180/5), color = 'darkblue', 
             hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth': 4})

# interaction effects block:condition
sns.distplot(z_cond_block_int_reciplab['z_condition[T.treatment]:C(blockno)[T.2]'], hist=True, kde=True, 
             bins=int(180/5), color = 'darkblue', 
             hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth': 4},
             axlabel = 'z-parameter treatment:block2')
sns.distplot(z_cond_block_int_reciplab['z_condition[T.treatment]:C(blockno)[T.3]'], hist=True, kde=True, 
             bins=int(180/5), color = 'darkblue', 
             hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth': 4},
             axlabel = 'z-parameter treatment:block3')
sns.distplot(z_cond_block_int_reciplab['z_condition[T.treatment]:C(blockno)[T.3]']-z_cond_block_int_reciplab['z_condition[T.treatment]:C(blockno)[T.2]'], hist=True, kde=True, 
             bins=int(180/5), color = 'darkblue', 
             hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth': 4},
             axlabel = 'z-parameter treatment:block3-block2')

int_t2_t3_recip = list((z_cond_block_int_reciplab['z_condition[T.treatment]:C(blockno)[T.2]']-z_cond_block_int_reciplab['z_condition[T.treatment]:C(blockno)[T.3]'])>0)
int_t2_t3_recip.count(True)
int_t1_t2_recip = list((z_cond_block_int_reciplab['z_condition[T.treatment]:C(blockno)[T.2]'])>0)
int_t1_t2_recip.count(True)

##### hypothesis testing

z_cond_blockno_int = m_recip_reg_cond_z_int_p05.nodes_db.node[['z_condition[T.treatment]:blockno']]
print "P_z_int = ", (z_cond_blockno_int.trace() > 0).mean()
