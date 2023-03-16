import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

def figure1(ki_df):
    plt.rcParams["figure.facecolor"] = "w"
    plt.rcParams['figure.figsize'] = [10, 7]
    g=sns.heatmap(ki_df, cmap='rocket_r', cbar_kws={'label': 'pKi'})
    g.set_facecolor('gray')
    plt.ylabel('Antipsychotic')
    plt.xlabel('Receptor')
    plt.savefig('./results/figure1.pdf', bbox_inches='tight', dpi=400)


def figure2(ki_corr, ki_df_scaled, cluster_idxs, shifts):
    labels = ki_df_scaled.index[cluster_idxs]
    data = pd.DataFrame(ki_corr[cluster_idxs,:][:,cluster_idxs], index = labels, columns=labels)
    plt.rcParams['figure.figsize'] = [6, 8]
    g = sns.heatmap(data, cmap='RdBu_r', center=0, cbar_kws={'label': '$r_{p}$', 'location':'top', 'orientation': 'horizontal'})
    plt.vlines(shifts,ymin=0, ymax=len(labels))
    plt.hlines(shifts,xmin=0, xmax=len(labels))
    g.set_xticklabels(labels=g.get_xticklabels(), rotation = 45, ha='right')
    plt.savefig('./results/figure2.pdf', bbox_inches='tight', dpi=400)


def figure3(X_df, ki_df, clusters, n_comp, pca_summary):
    X_df.columns=['1', '2', '3']
    plt.rcParams['figure.figsize'] = [12, 10]
    fig=plt.figure()
    sns.set_style('white')
    gs = GridSpec(4, 4, left=0.5, right=0.98, hspace=0.15, wspace=0.9)
    axes = {}
    for i in range(4):
        axes[f'ax{i+1}a']=fig.add_subplot(gs[i,0]) # i row, first column
        axes[f'ax{i+1}']=fig.add_subplot(gs[i,1]) # i row, second column
    axes['ax5']=fig.add_subplot(gs[:,2:]) # Second col, span all rows
    
    # Barplots
    for i in range(4):
        ax = sns.barplot(x=np.arange(1, n_comp+1), y=pca_summary[:,i], ax=axes[f'ax{i+1}'])
        ax.set(ylim=(-0.3,0.3))
        if i!=3:
            ax.set(xticklabels=[])
        else:
            for i, tick_label in enumerate(ax.get_xticklabels()):
                    if i==0:
                        tick_label.set_color("blue")
                    if i==1:
                        tick_label.set_color("orange")
                    if i==2:
                        tick_label.set_color("green")
    # Heatmap
    sns.heatmap(X_df, cmap='RdBu_r', center=0, ax=axes['ax5'], cbar_kws={'label': 'PCA loading', 'location':'right'})
    axes['ax5'].set_ylabel('')
    for i, tick_label in enumerate(axes['ax5'].get_xticklabels()):
        if i==0:
            tick_label.set_color("blue")
        if i==1:
            tick_label.set_color("orange")
        if i==2:
            tick_label.set_color("green")
    # text
    group_names = ['Muscarinic', "Adrenergic/Low DA ", "Serotonergic/Dopaminergic", "Dopaminergic"]
    for h in range(4):
        for i, ap in enumerate(ki_df.index[clusters[f'{h}']]):
            axes[f'ax{h+1}a'].text(0.15,0.8-0.1*i, ap)
        axes[f'ax{h+1}a'].text(-0.2,0.5, f'Cluster {h+1}', rotation=90, va='center', weight='bold')
        axes[f'ax{h+1}a'].text(-0.05,0.5, group_names[h], rotation=90, va='center', style='italic')
        axes[f'ax{h+1}a'].xaxis.set_major_locator(ticker.NullLocator())
        axes[f'ax{h+1}a'].yaxis.set_major_locator(ticker.NullLocator())
    sns.despine(bottom = True, left = True)
    plt.savefig('./results/figure3.pdf', bbox_inches='tight', dpi=400)


def figure4(mean_ses, pca_se_df):
    plt.rcParams['figure.figsize'] = [8, 6]
    fig, (ax1, ax2)= plt.subplots(nrows=2, ncols=1)
    sns.heatmap( -1*pca_se_df.T, cmap='RdBu_r', center=0, cbar_kws={'label': '$r_{p}$'},ax=ax1)
    ax1.set_yticklabels(['PC1', 'PC2', 'PC3'], rotation = 0)
    for i, tick_label in enumerate(ax1.get_yticklabels()):
        if i==0:
            tick_label.set_color("blue")
        if i==1:
            tick_label.set_color("orange")
        if i==2:
            tick_label.set_color("green")
    labels = [x.capitalize().replace('_', ' ')[:-7] for x in pca_se_df.index]
    ax1.set_xticklabels(labels,rotation = 45, ha='right')
    ax1.set_title('A', loc='left')
    sns.heatmap(mean_ses.values, cmap='rocket_r', cbar_kws={'label': 'weight'}, ax=ax2)
    ax2.set_yticklabels(['Cluster 1\n Muscarinic', 'Cluster 2\n Adrenergic/Low DA', 'Cluster 3\n Serotonergic/Dopaminergic', 'Cluster 4\n Dopaminergic'], rotation = 0)
    labels = [x.capitalize().replace('_', ' ')[:-7] for x in mean_ses.columns]
    ax2.set_xticklabels(labels,rotation = 45, ha='right')
    ax2.set_title('B', loc='left')
    plt.tight_layout()   
    plt.savefig('./results/figure4.pdf', bbox_inches='tight', dpi=400)



def figure5a(ki_df, ax):
    ki_df_summary = pd.DataFrame(index = ki_df.index)
    for method in ['typical', 'nbn_grouping', 'new_grouping']:
        ki_df_new = pd.read_csv(f'./data/{method}.csv', delimiter=',').set_index('antipsychotic')
        ki_df_new['summary'] = 0
        for i, col in enumerate(ki_df_new.columns[:-1]):
            ki_df_new['summary']  = np.where(ki_df_new[col]==1, i+1,  ki_df_new['summary'])
        if method == 'new_grouping':
            ki_df_new['summary'].loc[ki_df_new['summary']==0]=4
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
    [' Partial agonist \n Receptor partial agonist (D2, 5-HT1A) \n Cluster 2: Adrenergic/Low DA', 
    ' Atypical \n Receptor Antagonist (D2, 5-HT2) \n Cluster 1: Muscarinic', 
    ' Typical \n  Receptor Antagonist (D2) \n Cluster 4: Dopaminergic',  
    ' Receptor Antagonist (D2, 5-HT2, NEa2) \n Cluster 3: Serotonergic/Dopaminergic', 
    ' Receptor antagonist (5-HT2, D2, NEa2)',
    ' Receptor antagonist (D2, 5-HT2) + reuptake inhibitor (NET)',
    ' Receptor antagonist (5-HT2, D2)'])
    idx=[0,2,4,1,3,5,6]
    value_to_int = {j:i for i,j in enumerate(pd.unique(ki_df_summary.values.ravel())[np.array(idx)])} 
    n = len(value_to_int)
    plt.rcParams['figure.figsize'] = [7, 5]
    cmap = sns.color_palette("cubehelix", n) 
    ax = sns.heatmap(ki_df_summary.replace(value_to_int).sort_values(['Typical/Atypical', 'NBN']), cmap=cmap, ax=ax)
    ax.invert_yaxis()    
    colorbar = ax.collections[0].colorbar 
    r = colorbar.vmax - colorbar.vmin 
    colorbar.set_ticks([colorbar.vmin + r / n * (0.5 + i) for i in range(n)])
    colorbar.set_ticklabels(list(value_to_int.keys()))                                
    plt.ylabel('')
    plt.tight_layout()
    

def figure5b(obs_results, perm_results,ax):
    labels = {'new_grp': 'Data Driven Grouping',  'typical': 'Typical/Atypical grouping', 'recep': 'Full Receptor profile', 'nbn': 'NBN grouping',}
    for analysis, color in zip([ 'new_grp', 'typical', 'recep','nbn'], ['xkcd:cornflower blue', 'orange', 'g', 'r']):
        p = np.sum(perm_results[f'{analysis}']<obs_results[f'{analysis}'])/len(perm_results[f'{analysis}'])
        print(p)
        if p<0.05:
            plt.text(obs_results[f'{analysis}']-0.005, 2.5,'*', color=color)
        sns.kdeplot(perm_results[f'{analysis}'], label = labels[analysis],color=color, ax=ax)
        plt.vlines(obs_results[f'{analysis}'], 0, 2, color=color)
        plt.legend(framealpha=1, frameon=False)
    plt.xlabel('Median Error')
    plt.ylabel('Frequency')
    sns.despine()
    plt.tight_layout()