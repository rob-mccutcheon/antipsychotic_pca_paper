import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.ticker as ticker
import numpy as np

def figure3(X_df, ki_df, clusters, n_comp, pca_summary):
    X_df.columns=['1', '2', '3']
    fig=plt.figure()
    plt.rcParams['figure.figsize'] = [12, 10]
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
    plt.savefig('../results/figures/Figure3.png', bbox_inches='tight', dpi=400)
