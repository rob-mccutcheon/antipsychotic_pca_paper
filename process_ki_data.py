import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from funcs import datasorting as ds
from funcs import metrics as ms
from funcs import figures as fs
from funcs import pls as pls
from sklearn.preprocessing import StandardScaler
import pickle
import copy

# Get ki data and scale
ki_df = ds.get_ki_data()
ki_df = -np.log10(ki_df/1000000000) #convert to pki
ki_df.where(((ki_df>4) | (np.isnan(ki_df))), 4, inplace=True) # make the floor 4
ki_df = ki_df.loc[:,np.std(ki_df)!=0] #remove receptors with no info

# Figure 1: pKi summary
plt.rcParams['figure.figsize'] = [10, 7]
g=sns.heatmap(ki_df, cmap='rocket_r', cbar_kws={'label': 'pKi'})
g.set_facecolor('gray')
plt.ylabel('Antipsychotic')
plt.xlabel('Receptor')
plt.savefig('../results/figures/figure1.png', bbox_inches='tight', dpi=400)

# Probabilistic PCA
ki_df = ki_df-4 # set floor to zero
ki_df = ds.reverse_agonists(ki_df)
scaled_values = StandardScaler().fit_transform(ki_df)
n_comp = np.max(scaled_values.shape)
np.random.seed(1)
C, ss, M, X, Ye = pls.ppca(scaled_values.T, d=n_comp, dia=False)
nan_idx = np.where(np.isnan(scaled_values))
scaled_values[nan_idx] = Ye.T[nan_idx]
ki_df_scaled = pd.DataFrame(data=scaled_values, index = ki_df.index, columns=ki_df.columns)

# Clustering
ki_corr = ki_df_scaled.T.corr(method='pearson').values
gamma=1
cluster_idxs, shifts = ms.get_clusters(ki_corr, gamma)

# Figure 2: Plot clustered  pKi heatmap
labels = ki_df_scaled.index[cluster_idxs]
data = pd.DataFrame(ki_corr[cluster_idxs,:][:,cluster_idxs], index = labels, columns=labels)
plt.rcParams['figure.figsize'] = [6, 8]
g = sns.heatmap(data, cmap='RdBu_r', center=0, cbar_kws={'label': '$r_{p}$', 'location':'top', 'orientation': 'horizontal'})
plt.vlines(shifts,ymin=0, ymax=len(labels))
plt.hlines(shifts,xmin=0, xmax=len(labels))
g.set_xticklabels(labels=g.get_xticklabels(), rotation = 45, ha='right')
plt.savefig('../results/figures/figure2.png', bbox_inches='tight', dpi=400)

# Getting summary pca values for the clusters
clusters = {}
for i in np.arange(len(shifts)+1):
    if i == 0:
        clusters[f'{i}'] = cluster_idxs[:shifts[0]]
    elif i == len(shifts):
        clusters[f'{i}'] = cluster_idxs[shifts[i-1]:]
    else:
        clusters[f'{i}'] = cluster_idxs[shifts[i-1]:shifts[i]]

n_comp = 3
pca_summary = np.empty([n_comp,len(clusters)])
for comp in range(n_comp):
    i = 0
    for keys, cluster in clusters.items():
        pca_summary[comp, i] = np.mean(C[cluster,comp])
        i+=1

# Figure 3 - loadings for clusters
X_df = pd.DataFrame(X[:,:3], index = ki_df.columns)
X_df = X_df.sort_values(0)
fs.figure3(X_df*-1, ki_df, clusters, n_comp, pca_summary*-1)


# SE clusters relating to
# Get se data and scale
se_df = ds.get_se_data()
# Drop drugs without adequate receptor data
se_df = se_df.drop(index = ['Clopenthixol', 'Levomepromazine', 'Perazine', 'Zuclopenthixol', 'Penfluridol'])
se_df.loc[:,['totalsymptoms_effect', 'positive_symptoms_effect', 'negative_symptoms_effect']] = -1*se_df.loc[:,['totalsymptoms_effect', 'positive_symptoms_effect', 'negative_symptoms_effect']]
scaled_values = StandardScaler().fit_transform(se_df)


#ppca to impute missing side effect values
n_comp = np.max(scaled_values.shape)
np.random.seed(1)
C2, ss, M, X2, Ye2 = pls.ppca(scaled_values.T, d=n_comp, dia=False)
se_df_scaled = pd.DataFrame(data=Ye2.T, index = se_df.index, columns=se_df.columns)

# Correlation of PCA loadings with side effects 
X_df = pd.DataFrame(C[:,:3], index = ki_df.index)
merged_df = pd.merge(X_df, se_df_scaled, left_index=True, right_index=True)
pca_se_df = merged_df.corr().iloc[:3,3:].T
pca_se_df = pca_se_df.sort_values(0)
plt.rcParams['figure.figsize'] = [8, 3]
fig, ax = plt.subplots()
pca_se_df = pca_se_df.rename(index={'negsymptoms_effect': 'Negative Symptoms_effect', 
                                    'possymptoms_effect': 'Positive Symptoms_effect',
                                    'totalsymptoms_effect': 'Total Symptoms_effect'})
sns.heatmap(-1*pca_se_df.T, cmap='RdBu_r', center=0, cbar_kws={'label': '$r_{p}$'})
ax.set_yticklabels(['PC1', 'PC2', 'PC3'], rotation = 0)
for i, tick_label in enumerate(ax.get_yticklabels()):
    if i==0:
        tick_label.set_color("blue")
    if i==1:
        tick_label.set_color("orange")
    if i==2:
        tick_label.set_color("green")
labels = [x.capitalize().replace('_', ' ')[:-7] for x in pca_se_df.index]
ax.set_xticklabels(labels,rotation = 45, ha='right')
plt.savefig('../results/figures/figure4a.png', bbox_inches='tight', dpi=400)

# Getting mean side effect scores of clusters
merged_df2 = merged_df.reset_index()
merged_df2.loc[:,'weight_gain_effect':]=StandardScaler().fit_transform(se_df.values)
newDict = {}
for key,value in clusters.items():
    for val in value:
        if val in newDict:
            newDict[val].append(key)
        else:
            newDict[val] = key
merged_df2['clusters'] = pd.Series(newDict)
merged_df2 = merged_df2.set_index('index')


mean_ses = pd.DataFrame(index=clusters.keys(), columns = merged_df.loc[:,'weight_gain_effect':].columns)
for key, cluster_idx in clusters.items():
    for se in merged_df2.loc[:,'weight_gain_effect':].columns[:-1]:
        drugs = ki_df.index[cluster_idx]
        good_drugs = merged_df2.index.intersection(drugs)
        mean_ses.loc[key, se] = np.nanmean(merged_df2.loc[good_drugs, se])

mean_ses = mean_ses.astype(float)
mean_ses = mean_ses.rename(columns={'negsymptoms_effect': 'Negative Symptoms_effect', 
                                    'possymptoms_effect': 'Positive Symptoms_effect',
                                    'totalsymptoms_effect': 'Total Symptoms_effect'})
try:        
    mean_ses = mean_ses.drop(columns='clusters')
except KeyError:
    pass
plt.rcParams['figure.figsize'] = [8, 3]
fig, ax = plt.subplots()
mean_ses = mean_ses.T.sort_values('0').T
sns.heatmap(mean_ses.values, cmap='rocket_r', cbar_kws={'label': 'weight'})
ax.set_yticklabels(['Cluster 1\n Muscarinic', 'Cluster 2\n Adrenergic/Low DA', 'Cluster 3\n Serotonergic/Dopaminergic', 'Cluster 4\n Dopaminergic'], rotation = 0)
labels = [x.capitalize().replace('_', ' ')[:-7] for x in mean_ses.columns]
ax.set_xticklabels(labels,rotation = 45, ha='right')   
plt.savefig('../results/figures/figure4b.png', bbox_inches='tight', dpi=400)


# PLS predictions: Receptor profile, groupings, NBN
perm_results = {'recep_corr':[], 'new_grp_corr':[], 'nbn_corr':[], 'typical_corr':[], 'se_corr':[], 'test_corr':[],
'recep_sme':[], 'new_grp_sme':[], 'nbn_sme':[], 'typical_sme':[], 'se_sme':[], 'test_sme':[]
}
obs_results = {'recep_corr':[], 'new_grp_corr':[], 'nbn_corr':[], 'typical_corr':[], 'se_corr':[], 'test_corr':[],
'recep_sme':[], 'new_grp_sme':[], 'nbn_sme':[], 'typical_sme':[], 'se_sme':[], 'test_sme':[]
}

se_df = ds.get_se_data()
se_df = se_df.drop(index = ['Clopenthixol', 'Levomepromazine', 'Perazine', 'Zuclopenthixol', 'Penfluridol'])
se_df.loc[:,['totalsymptoms_effect', 'positive_symptoms_effect', 'negative_symptoms_effect']] = -1*se_df.loc[:,['totalsymptoms_effect', 'positive_symptoms_effect', 'negative_symptoms_effect']]
se_df = se_df - np.nanmin(se_df, axis=0)

sumy = []
num_perms = 500
for analysis in ['recep', 'new_grp', 'nbn', 'typical']:#, 'se', 'test']:
    if analysis == 'recep':
        ki_df = ds.get_ki_data()
        ki_df = -np.log10(ki_df/1000000000) #convert to pki
        ki_df.where(((ki_df>4) | (np.isnan(ki_df))), 4, inplace=True) # make the floor 4
        ki_df = ki_df.loc[:,np.std(ki_df)!=0] #remove receptors with no info
        ki_df_scaled = StandardScaler().fit(ki_df)
        ki_df = ki_df-4 # set floor to zero
        ki_df = ds.reverse_agonists(ki_df)
        scaled_values = StandardScaler().fit_transform(ki_df)
        n_comp = np.max(scaled_values.shape)
        np.random.seed(1)
        C, ss, M, X, Ye = pls.ppca(scaled_values.T, d=n_comp, dia=False)
        ki_df = pd.DataFrame(data=Ye.T, index = ki_df.index, columns=ki_df.columns)

    if analysis == 'new_grp':
        ki_df = pd.read_csv('../data/groupings.csv', delimiter=',').set_index('antipsychotic')

    if analysis == 'nbn':
        ki_df = pd.read_csv('../data/nbn_grouping.csv', delimiter=',').set_index('antipsychotic')

    if analysis == 'typical':
        ki_df = pd.read_csv('../data/typical.csv', delimiter=',').set_index('antipsychotic')

    if analysis == 'se':
        ki_df = pd.read_csv('../data/se.csv', delimiter=',').set_index('antipsychotic')

    merged_df = pd.merge(ki_df, se_df, left_index=True, right_index=True)
    num_grp = len(ki_df.T)
    shuffled_df = copy.deepcopy(merged_df)   
    obs_res = pls.get_scores(merged_df, num_grp)
    obs_results[f'{analysis}_corr'] = obs_res[0]
    obs_results[f'{analysis}_sme'] = obs_res[1]

    for h in range(num_perms):
        print(f'{analysis}{h}')
        shuffled_grp = shuffled_df.iloc[:,:num_grp]

        np.random.seed(h)
        np.random.shuffle(shuffled_grp.values)
        shuffled_df = pd.merge(shuffled_grp, shuffled_df.iloc[:, num_grp:], left_index=True, right_index=True)
        res = pls.get_scores(shuffled_df, num_grp)
        perm_results[f'{analysis}_corr'].append(res[0])
        perm_results[f'{analysis}_sme'].append(res[1])

# with open('../results/saved_data/perm_results.pkl', 'wb') as handle:
#     pickle.dump(perm_results, handle)
# with open('../results/saved_data/obs_results.pkl', 'wb') as handle:
#     pickle.dump(obs_results, handle)

plt.rcParams['figure.figsize'] = [6, 6]
obs_results = pd.read_pickle('../results/saved_data/obs_results.pkl')
perm_results = pd.read_pickle('../results/saved_data/perm_results.pkl')
labels = {'new_grp': 'Data Driven Grouping',  'typical': 'Typical/Atypical grouping', 'recep': 'Full Receptor profile', 'nbn': 'NBN grouping',}

plt.rcParams['figure.figsize'] = [6, 4]
for analysis, color in zip([ 'new_grp', 'typical', 'recep','nbn'], ['xkcd:cornflower blue', 'orange', 'g', 'r']):
    p = np.sum(perm_results[f'{analysis}_sme']<obs_results[f'{analysis}_sme'])/len(perm_results[f'{analysis}_sme'])
    print(p)
    if p<0.05:
        plt.text(obs_results[f'{analysis}_sme']-0.005, 2.5,'*', color=color)

    sns.kdeplot(perm_results[f'{analysis}_sme'], label = labels[analysis],color=color)
    plt.vlines(obs_results[f'{analysis}_sme'], 0, 2, color=color)
    plt.legend(framealpha=1, frameon=False)
plt.xlabel('Median Error')
plt.ylabel('Frequency')
sns.despine()
plt.tight_layout()
plt.savefig('../results/figures/figure5b.png', bbox_inches='tight', dpi=400)


# Illustrate classification scheme
ki_df_summary = pd.DataFrame(index = ki_df.index)
for method in ['typical', 'nbn_grouping', 'groupings']:
    ki_df_new = pd.read_csv(f'../data/{method}.csv', delimiter=',').set_index('antipsychotic')
    ki_df_new['summary'] = 0
    for i, col in enumerate(ki_df_new.columns[:-1]):
        ki_df_new['summary']  = np.where(ki_df_new[col]==1, i+1,  ki_df_new['summary'])
    ki_df_summary[method] = ki_df_new['summary']
ki_df_summary.columns = [ 'Typical/Atypical', 'NBN','Receptor based']

ki_df_summary['NBN']  = np.where(ki_df_summary['NBN']==6, 8,  ki_df_summary['NBN'])
ki_df_summary['NBN']  = np.where(ki_df_summary['NBN']==4, 6,  ki_df_summary['NBN'])
ki_df_summary['NBN']  = np.where(ki_df_summary['NBN']==3, 4,  ki_df_summary['NBN'])
ki_df_summary['NBN']  = np.where(ki_df_summary['NBN']==2, 3,  ki_df_summary['NBN'])
ki_df_summary['NBN']  = np.where(ki_df_summary['NBN']==8, 2,  ki_df_summary['NBN'])

ki_df_summary['Receptor based']  = np.where(ki_df_summary['Receptor based']==2, 8,  ki_df_summary['Receptor based'])
ki_df_summary['Receptor based']  = np.where(ki_df_summary['Receptor based']==1, 2,  ki_df_summary['Receptor based'])
ki_df_summary['Receptor based']  = np.where(ki_df_summary['Receptor based']==8, 1,  ki_df_summary['Receptor based'])
ki_df_summary = ki_df_summary.astype(int)

ki_df_summary = ki_df_summary.replace([1,2,3,4,5,6,7], 
[' Partial agonist \n Receptor partial agonist (D2, 5-HT1A) \n Cluster 2', 
' Atypical \n Receptor Antagonist (D2, 5-HT2) \n Cluster 1', 
' Typical \n  Receptor Antagonist (D2) \n Cluster 3',  
' Receptor Antagonist (D2, 5-HT2, NEa2) \n Cluster 4', 
' Receptor antagonist (5-HT2, D2, NEa2)',
' Receptor antagonist (D2, 5-HT2) + reuptake inhibitor (NET)',
' Receptor antagonist (5-HT2, D2)'])
idx=[3,1,0, 2,4,5,6]
value_to_int = {j:i for i,j in enumerate(pd.unique(ki_df_summary.values.ravel())[np.array(idx)])} # like you did
n = len(value_to_int)

plt.rcParams['figure.figsize'] = [5, 7]
cmap = sns.color_palette("cubehelix", n) 
ax = sns.heatmap(ki_df_summary.replace(value_to_int).sort_values(['Typical/Atypical', 'NBN']), cmap=cmap)
ax.invert_yaxis()    
colorbar = ax.collections[0].colorbar 
r = colorbar.vmax - colorbar.vmin 
colorbar.set_ticks([colorbar.vmin + r / n * (0.5 + i) for i in range(n)])
colorbar.set_ticklabels(list(value_to_int.keys()))                                
plt.ylabel('')
plt.savefig('../results/figures/figure5a.png', bbox_inches='tight', dpi=400)