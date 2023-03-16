import pandas as pd
import numpy as np

def reverse_agonists(ki_df):
    ki_df.loc[['Aripiprazole', 'Brexpiprazole', 'Cariprazine'], 'DRD2']  = -1*ki_df.loc[['Aripiprazole', 'Brexpiprazole', 'Cariprazine'], 'DRD2']
    ki_df.loc[['Aripiprazole', 'Brexpiprazole', 'Cariprazine'], 'DRD3']  = -1*ki_df.loc[['Aripiprazole', 'Brexpiprazole', 'Cariprazine'], 'DRD2']
    ki_df.loc[['Clozapine', 'Quetiapine', 'Asenapine', 'Aripiprazole', 'Brexpiprazole', 'Cariprazine', 'Lurasidone', 'Ziprasidone'], 'HTR1A']  = -1*ki_df.loc[['Clozapine', 'Quetiapine', 'Asenapine', 'Aripiprazole', 'Brexpiprazole', 'Cariprazine', 'Lurasidone', 'Ziprasidone'], 'HTR1A']
    ki_df.loc[['Olanzapine', 'Clozapine', 'Asenapine', 'Aripiprazole', 'Brexpiprazole', 'Ziprasidone'], 'HTR1B']  = -1*ki_df.loc[['Olanzapine', 'Clozapine', 'Asenapine', 'Aripiprazole', 'Brexpiprazole', 'Ziprasidone'], 'HTR1B'] 
    ki_df.loc[['Aripiprazole'], 'HTR2A']  = -1*ki_df.loc[['Aripiprazole'], 'HTR2A']
    ki_df.loc[['Aripiprazole'], 'HTR2C']  = -1*ki_df.loc[['Aripiprazole'], 'HTR2C']
    ki_df.loc[['Aripiprazole'], 'HTR6']  = -1*ki_df.loc[['Aripiprazole'], 'HTR6']
    ki_df.loc[['Aripiprazole'], 'HTR7']  = -1*ki_df.loc[['Aripiprazole'], 'HTR7']
    ki_df.loc[['Clozapine', 'Olanzapine'], 'CHRM1']  = -1*ki_df.loc[['Clozapine', 'Olanzapine'], 'CHRM1']
    return ki_df

def save_grouping(ki_df, ki_df_scaled, cluster_idxs, shifts):
    labels = ki_df_scaled.index[cluster_idxs]
    pls_aps = ['Amisulpride', 'Aripiprazole', 'Asenapine', 'Brexpiprazole', 'Cariprazine', 'Chlorpromazine', 
            'Clozapine', 'Flupentixol', 'Fluphenazine', 'Haloperidol', 'Iloperidone', 'Loxapine', 'Lurasidone', 
            'Molindone', 'Olanzapine', 'Paliperidone', 'Perphenazine', 'Pimozide', 'Quetiapine', 'Risperidone', 
            'Sertindole', 'Sulpiride', 'Thioridazine', 'Thiothixene', 'Trifluoperazine', 'Ziprasidone', 'Zotepine']
    group_df = pd.DataFrame(index = pls_aps)
    for i in range(len(shifts)):
        if i==0:
            group = labels[:shifts[i]]
            group_df[f'grp{i}'] = [ap in np.array(group) for ap in pls_aps] 
        elif i == len(shifts)-1:
            group = labels[shifts[i]:]
            group_df[f'grp{i}'] = [ap in np.array(group) for ap in pls_aps]
        else:
            group = labels[shifts[i-1]:shifts[i]]
            group_df[f'grp{i}'] = [ap in np.array(group) for ap in pls_aps]

    group_df.replace({False: 0, True: 1}, inplace=True)
    group_df.index.rename('antipsychotic', inplace=True)
    group_df.to_csv('./data/new_grouping.csv')
