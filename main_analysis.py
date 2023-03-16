import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from funcs import metrics as ms
from funcs import figures as fs
from funcs import data_sorting as ds
from sklearn.preprocessing import StandardScaler
import pickle

# Get ki data and scale
ki_df = pickle.load(open('./data/kis.pkl', 'rb'))
ki_df = -np.log10(ki_df/1000000000) #convert to pki
ki_df.where(((ki_df>4) | (np.isnan(ki_df))), 4, inplace=True) # make the floor 4
ki_df = ki_df.loc[:,np.std(ki_df)!=0] #remove receptors with no info

# Figure 1: pKi summary
fs.figure1(ki_df)

# Probabilistic PCA
ki_df = ki_df-4 # set floor to zero
ki_df = ds.reverse_agonists(ki_df)
scaled_values = StandardScaler().fit_transform(ki_df)
n_comp = np.max(scaled_values.shape)
np.random.seed(0)
C, ss, M, X, Ye = ms.ppca(scaled_values.T, d=n_comp, dia=False)
nan_idx = np.where(np.isnan(scaled_values))
scaled_values[nan_idx] = Ye.T[nan_idx]
ki_df_scaled = pd.DataFrame(data=scaled_values, index = ki_df.index, columns=ki_df.columns)

# Clustering
ki_corr = ki_df_scaled.T.corr(method='pearson').values
cluster_idxs, shifts = ms.get_clusters(ki_corr)

# Save csv for PLS analysis at the end
ds.save_grouping(ki_df, ki_df_scaled, cluster_idxs, shifts)

# Figure 2: Plot clustered  pKi heatmap
fs.figure2(ki_corr, ki_df_scaled, cluster_idxs, shifts)

# Figure 3 - PCA loadings for clusters
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
X_df = pd.DataFrame(X[:,:3], index = ki_df.columns)
X_df = X_df.sort_values(0)
fs.figure3(-1*X_df, ki_df, clusters, n_comp, -1*pca_summary)

# Figure 4: Correlation of PCA loadings and groupings with side effects 
se_df = pickle.load(open('./data/ses.pkl', 'rb'))
C_df = pd.DataFrame(C[:,:3], index = ki_df.index)
se_values = pd.DataFrame(StandardScaler().fit_transform(se_df.values), index=se_df.index, columns = se_df.columns)
merged_df = pd.merge(C_df, se_values, left_index=True, right_index=True)
pca_se_df = merged_df.corr().iloc[:3,3:].T
pca_se_df = pca_se_df.sort_values(0)
move_cols = ['negative_symptoms_effect','positive_symptoms_effect', 'totalsymptoms_effect']
pca_se_df = pca_se_df.drop(move_cols).append(pca_se_df.loc[move_cols])

mean_ses = pd.DataFrame(index=clusters.keys(), columns = merged_df.loc[:,'weight_gain_effect':].columns)
for key, cluster_idx in clusters.items():
    for se in merged_df.loc[:,'weight_gain_effect':].columns:
        drugs = ki_df.index[cluster_idx]
        good_drugs = merged_df.index.intersection(drugs)
        mean_ses.loc[key, se] = np.nanmean(merged_df.loc[good_drugs, se])
mean_ses = mean_ses.T.sort_values('0').T
mean_ses = mean_ses.astype(float)
mean_ses = mean_ses.T.drop(move_cols).append(mean_ses.T.loc[move_cols]).T
fs.figure4(mean_ses, pca_se_df)

# Figure 5: PLS predictions
obs_results = {'recep':[], 'new_grp':[], 'nbn':[], 'typical':[]}
perm_results = {'recep':[], 'new_grp':[], 'nbn':[], 'typical':[]}

num_perms = 1000
for analysis in obs_results.keys():
    if analysis == 'recep': grp_df = ki_df_scaled
    if analysis == 'new_grp': grp_df = pd.read_csv('./data/new_grouping.csv', delimiter=',').set_index('antipsychotic')
    if analysis == 'nbn': grp_df = pd.read_csv('./data/nbn_grouping.csv', delimiter=',').set_index('antipsychotic')
    if analysis == 'typical':grp_df = pd.read_csv('./data/typical.csv', delimiter=',').set_index('antipsychotic')
    merged_df = pd.merge(grp_df, se_df, left_index=True, right_index=True)
    num_grp = len(grp_df.T)
    shuffled_df = copy.deepcopy(merged_df)   
    obs_results[f'{analysis}'] = ms.get_scores(merged_df, num_grp)
    for h in range(num_perms):
        print(f'{analysis}{h}')
        shuffled_grp = shuffled_df.iloc[:,:num_grp]
        np.random.seed(h)
        np.random.shuffle(shuffled_grp.values)
        shuffled_df = pd.merge(shuffled_grp, shuffled_df.iloc[:, num_grp:], left_index=True, right_index=True)
        res = ms.get_scores(shuffled_df, num_grp)
        perm_results[f'{analysis}'].append(res)

pickle.dump(perm_results, open('./results/perm_results.pkl', 'wb'))
pickle.dump(obs_results, open('./results/obs_results.pkl', 'wb'))
perm_results = pickle.load(open('./results/perm_results.pkl', 'rb'))
obs_results = pickle.load(open('./results/obs_results.pkl', 'rb'))

#P values
for analysis in perm_results.keys():
    p = np.sum(obs_results[analysis]>=perm_results[analysis])/1000
    print(f'{analysis} {p}')


# Join figures 5a and b
plt.rcParams['figure.figsize'] = [10, 10]
fig, (ax1, ax2)= plt.subplots(nrows=2, ncols=1, height_ratios=[1.6,1])
fs.figure5a(ki_df, ax1)
fs.figure5b(obs_results, perm_results, ax2)
ax1.set_title('A', loc='left')
ax2.set_title('B', loc='left')
fig.tight_layout()
plt.savefig('./results/figure5.pdf', bbox_inches='tight', dpi=400)