import pandas as pd
import numpy as np

def get_se_data():
    se_data = pd.read_csv('/Users/robmcc/Documents/academic/antipsychotic_PCA/data/se_current.csv')

    se_data = se_data[se_data['broad_class'] == 'antipsychotic']
    se_data['med_name'] = se_data['med_name'].str.capitalize()
    columns = ['med_name', 'weight_gain_effect',
       'parkinsonism_effect', 'akathisia_effect',
       'prolactin_effect', 'QTc_effect', 'sedation_effect',
       'anticholinergic_effects_effect', 'tardive_dyskinesia_effect',
       'dystonia_effect', 'seizures_effect', 'postural_hypotension_effect',
       'hyperlipidemia_effect', 'hyperglycaemia_effect', 'totalsymptoms_effect','positive_symptoms_effect','negative_symptoms_effect' ]
    se_data = se_data.loc[:,columns]
    se_data.rename(columns={'med_name':'antipsychotic', 'QTc_effect':'QTc_prolongation_effect'}, inplace=True)
    se_data = se_data.set_index('antipsychotic')
    se_data.rename(index={'Thiotixene': 'Thiothixene'}, inplace=True)
    se_data = se_data.astype(float)

    # A SE should be present in at least 5 drugs to be included
    drop = []
    for rec in se_data.columns:
        if np.sum(~np.isnan(se_data.loc[:,rec]))<5:
            drop.append(rec)
    se_data = se_data.drop(columns=drop)
            
     # A drug has to have at least 5 ses to be included
    drop = []
    for drug in se_data.index:
        if np.sum(~np.isnan(se_data.loc[drug,: ]))<5:
            drop.append(drug)
    se_data = se_data.drop(index=drop)
    
    return se_data


def get_ki_data():
    ki_df = pd.read_csv('../data/KiDatabase.csv', delimiter=',', encoding='ISO 8859-1')
    drugs = ['Amisulpride', 'ARIPIPRAZOLE', 'ASENAPINE', 'brexpiprazole', 'cariprazine', 'clozapine',  'clopenthixol',  'chlorpromazine',  'fluphenazine',  'flupenthixol',  'haloperidol',  'iloperidone',  'levomepromazine',  'loxapine',  'lurasidone',   'molindone',  'olanzapine',  'paliperidone',  'penfluridol', 'perazine',  'perphenazine',  'pimozide',  'quetiapine',  'risperidone',  'sertindole',  'sulpiride',   'thioridazine',  'thiothixene',  'trifluoperazine',  'ziprasidone', 'zotepine', 'zuclopenthixol']
    drugs = [d.upper() for d in drugs]
    drugs = drugs + [d.capitalize() for d in drugs]
    drugs = drugs + [d.lower() for d in drugs]

    ki_df.loc[ki_df['Ligand Name']=='9-OH-risperidone', 'Ligand Name'] = 'Paliperidone'
    ki_df.loc[ki_df['Ligand Name']=='ORG-5222', 'Ligand Name'] = 'Asenapine'
    ki_df.loc[ki_df['Ligand Name']=='SM 13496', 'Ligand Name'] = 'Lurasidone'
    ki_df.loc[ki_df['Ligand Name']=='FLUPENTHIXOL, Alpha', 'Ligand Name'] = 'Flupenthixol'
    ki_df.loc[ki_df['Ligand Name']=='Flupenthixol, Beta', 'Ligand Name'] = 'Flupenthixol'
    ki_df.loc[ki_df['Ligand Name']=='FLUPENTHIXOL, CIS', 'Ligand Name'] = 'Flupenthixol'
    ki_df.loc[ki_df['Ligand Name']=='Flupenthixol, trans', 'Ligand Name'] = 'Flupenthixol'
    ki_df.loc[ki_df['Ligand Name']=='Flupenthixol, Trans-(E)', 'Ligand Name'] = 'Flupenthixol'
    ki_df.loc[ki_df['Ligand Name']=='Flupenthixol,a-', 'Ligand Name'] = 'Flupenthixol'
    ki_df.loc[ki_df['Ligand Name']=='cis-Clopenthixol', 'Ligand Name'] = 'Zuclopenthixol'

    # subset to only include antipsychotic and human studies
    ki_df = ki_df[ki_df['Ligand Name'].isin(drugs)]
    ki_df['Ligand Name'] = ki_df['Ligand Name'].str.capitalize()
    ki_df = ki_df[ki_df['species']=='HUMAN']
    drugs = np.sort(pd.unique(ki_df['Ligand Name']).astype(str))


    # replace unigene where names missing
    # translate 
    translator = {
     '5-HT1':'HTR1', #UNSPECIFIED
     '5-HT2':'HTR2',#UNSPECIFIED
     '5-HT2c VGI':'HTR2C',
     '5-HT2C-INI':'HTR2C',
     '5-HT7b':'HTR7',
     '5-HT7L':'HTR7',
     '5-HT7S':'HTR7',
     'alpha1A': 'ADRA1A',
     'alpha1B': 'ADRA1B',
     'Alpha1D': 'ADRA1D',
     'Alpha2A': 'ADRA2A',
     'Alpha2B': 'ADRA2B',
     'Alpha2C': 'ADRA2C',
     'beta1':'ADRB1',
     'beta2':'ADRB2',
     'beta3':'ADRB3',
     'alpha1': 'ADRA1', #UNSPECIFIED
     'alpha1-Adrenocepter': 'ADRA1', #UNSPECIFIED
     'Adrenaline alpha1': 'ADRA1', #UNSPECIFIED
     'adrenergic Alpha1': 'ADRA1', #UNSPECIFIED
     'adrenergic Alpha1A': 'ADRA1A', #UNSPECIFIED
     'adrenergic Alpha2': 'ADRA2', #UNSPECIFIED
     'alpha2': 'ADRA2',
     'alpha2-Adrenocepter': 'ADRA2',
     'Cholinergic, muscarinic':'CHRM',  #unspec
     'Cholinergic, Nicotinic Alpha1Beta2': 'CHRNA1B2',
     'Cholinergic, Nicotinic Alpha2Beta2': 'CHRNA2B2',
     'Cholinergic, Nicotinic Alpha2Beta4': 'CHRNA2B4',
     'Cholinergic, Nicotinic Alpha3Beta2': 'CHRNA3B2',
     'Cholinergic, Nicotinic Alpha3Beta4': 'CHRNA3B4',
     'Cholinergic, Nicotinic Alpha7': 'CHRNA7',
     'D1':'DRD1',
     'D2':'DRD2',
     'D2L':'DRD2',
     'D2S':'DRD2',
     'D3':'DRD3',
     'D4':'DRD4',
     'D5':'DRD5',
     'DAT':'SLC6A3',
     'Dopamine D1A': 'DRD1',
     'Dopamine D2A': 'DRD2',
     'DOPAMINE D4.2': 'DRD4',
     'DOPAMINE D4.4': 'DRD4',
     'Dopamine2-like':'DRD2',
     'H1':'HRH1',
     'H2':'HRH2',
     'H3':'HRH3',
     'H4':'HRH4',
     'h5-HT1A':'HTR1A',
     'h5-HT2A':'HTR2A',
     'h5-HT2B':'HTR2B',
     'halpha1B-adrenergic':'ADRA1B',
     'halpha2C-adrenergic':'ADRA2C',
     'hD2L': 'DRD2',
     'hD3': 'DRD3',
     'HERG': 'HERG',
     'hH1': 'HRH1',
     'hM1':'CHRM1',
     'Muscarinic':'CHRM',
     'Muscarinic M1': 'CHRM1',
     'noradrenaline-alpha1':'ADRA1',
     'noradrenaline-alpha2A':'ADRA2A',
     'noradrenaline-alpha2C':'ADRA2C',
     'Norepinephrine transporter': 'NAT',
     'Serotonin 5-HT1A':'HTR1A',
     'Serotonin 5-HT2A':'HTR2A',
     'Serotonin 5-HT2B':'HTR2B',
     'Serotonin 5-HT2C':'HTR2C',
     'Serotonin 5-HT7':'HTR7',
     'SERT':'SLC6A4',
     'Substance P': 'SUBP',
     'Vasopressin V3': 'V3'
     }
    
    for key,value in translator.items():
        print(key)
        ki_df.loc[ki_df['Name']==key, 'Unigene'] = value
    
    # a few random receptors (<5) that we do not incliude
    ki_df.dropna(subset=['Unigene'], how='all', inplace=True)

    receptors = sorted(pd.unique(ki_df['Unigene']).astype(str))


    df = pd.DataFrame(columns = ['drug', 'receptor', 'ki'])

    for drug in drugs:
        for receptor in receptors:
            sub_df = ki_df[ki_df['Ligand Name']==drug]
            sub_df = sub_df[sub_df['Unigene']==receptor]
            ki = np.median(sub_df['ki Val'])
            new_row = pd.DataFrame([[drug, receptor, ki]], columns = ['drug', 'receptor', 'ki'])
            df = pd.concat([df, new_row], axis=0)

    df = df.reset_index(drop=True)
    df = pd.pivot_table(df, index='drug', columns='receptor', values='ki')

    # df.reset_index(drop=True)
    df.index = [d.capitalize() for d in df.index]

    # A receptor should be present in at least 5 drugs to be included
    drop = []
    for rec in df.columns:
        if np.sum(~np.isnan(df.loc[:,rec]))<5:
            drop.append(rec)
    df = df.drop(columns=drop)
            
     # A drug has to have at least 5 kis to be included
    drop = []
    for drug in df.index:
        if np.sum(~np.isnan(df.loc[drug,: ]))<5:
            drop.append(drug)
    df = df.drop(index=drop)
    df.rename(index={'Flupenthixol': 'Flupentixol'}, inplace=True)
    return df


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