import pandas as pd
from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()

# Heart failure data
hf_data = pd.read_csv('data/heart_failure_clinical_records_dataset.csv')
hf_data_raw = hf_data.copy()

hf_data['sex'] = hf_data['sex'].astype('category')
for c in ['anaemia', 'diabetes', 'high_blood_pressure', 'smoking', 'DEATH_EVENT']:
    hf_data[c] = hf_data[c].astype('bool')

hf_data['sex'] = hf_data['sex'].map({1: 'Male', 0: 'Female'})

# Toxicity data
t_data = pd.read_csv('data/qsar_oral_toxicity.csv', sep=';', header=None)
t_data_raw = t_data.copy()

for c in t_data.columns[:-1]:
    t_data[c] = t_data[c].astype('bool')
t_data[t_data.columns[-1]] = t_data[t_data.columns[-1]].astype('category')
t_data.rename(columns={1024: 'toxic'}, inplace=True)
