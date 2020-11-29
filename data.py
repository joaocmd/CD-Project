import pandas as pd
from pandas.plotting import register_matplotlib_converters
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from imblearn.over_sampling import SMOTE

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
t_data.rename(columns={1024: 'toxic'}, inplace=True)
t_data_raw = t_data.copy()

for c in t_data.columns[:-1]:
    t_data[c] = t_data[c].astype('bool')
t_data[t_data.columns[-1]] = t_data[t_data.columns[-1]].astype('category')
t_data['toxic'].replace({'positive': True, 'negative': False}, inplace=True)

def get_hf_data(filter_outliers=False, feature_selection=False, scaling="none"):
    data = hf_data_raw.copy()
    if (filter_outliers):
        # creatinine_phosphokinase, outliers above 3000
        data = data[data["creatinine_phosphokinase"] <= 3000]

        # serum_creatinine, outliers above 4
        data = data[data["serum_creatinine"] <= 4]

        # platelets, outliers above 600000
        data = data[data["platelets"] <= 600000]
        
    if (feature_selection):
        data = data.drop(columns=['time'])
        
    df_nr = None
    df_sb = None
    target = data.pop('DEATH_EVENT')
    if (scaling != "none"):
        df_nr = pd.DataFrame(data, columns=data.select_dtypes(include=['float64','int64']).columns) 
        df_sb = pd.DataFrame(data, columns=data.select_dtypes(include=['bool']).columns)
        
    if (scaling == "z-score"):
        transf = StandardScaler(with_mean=True, with_std=True, copy=True).fit(df_nr)
        norm_data_zscore = pd.DataFrame(transf.transform(df_nr), columns=df_nr.columns)
        data = norm_data_zscore.join(df_sb, how='inner')
        
    if (scaling == "minmax"):
        transf = MinMaxScaler(feature_range=(0, 1), copy=True).fit(df_nr)
        norm_data_minmax = pd.DataFrame(transf.transform(df_nr), columns= df_nr.columns)
        data = norm_data_minmax.join(df_sb, how='inner')
    
    df_class_min = None
    df_class_max = None
    data['DEATH_EVENT'] = target
 
    return data

def get_t_data(feature_selection=0):
    data = t_data_raw.copy()
    
    def get_redundant_pairs(df, ratio):
        cols_to_drop = set()
        for i in range(0, df.shape[1]-1):
            if i in cols_to_drop:
                continue
            for j in range(i+1, df.shape[1]-1):
                if df[i][j] > ratio:
                    cols_to_drop.add(j)    
        return list(cols_to_drop)

    def remove_redundant_variables(df, ratio):
        au_corr = df.corr().abs()
        labels_to_drop = get_redundant_pairs(au_corr, ratio)
        print(df.drop(columns=labels_to_drop))
        return df.drop(columns=labels_to_drop)

    if feature_selection != 0:
        data = remove_redundant_variables(data, feature_selection)
        print(data)
        
    return data

def balance(data: pd.DataFrame, strat: str, target: str):
    unbal = data.copy()
    target_count = data[target].value_counts()
    min_class = target_count.idxmin()
    ind_min_class = target_count.index.get_loc(min_class)
    df_class_min = unbal[unbal[target] == min_class]
    df_class_max = unbal[unbal[target] != min_class]
     
    if (strat == "undersample"):
        res = pd.concat([df_class_max.sample(len(df_class_min)), df_class_min])
    
    if (strat == "oversample"):
        res = pd.concat([df_class_min.sample(len(df_class_max), replace=True), df_class_max])
        
    if (strat == "smote"):
        smote = SMOTE(sampling_strategy='minority', random_state=42069)
        y = unbal.pop(target).values
        X = unbal.values
        smote_X, smote_y = smote.fit_sample(X, y)
        res = pd.concat([pd.DataFrame(smote_X, columns=unbal.columns), pd.DataFrame(smote_y, columns=[target])], axis=1)
    
    return res

def get_corr(data, minimum_cor=0.95):
    df = data.copy()
    au_corr = df.corr().abs().unstack()
    labels_to_drop = set()
    cols = df.columns
    for i in range(0, df.shape[1]):
        for j in range(0, i+1):
            labels_to_drop.add((cols[i], cols[j]))
    most_cor = au_corr.drop(labels=labels_to_drop).sort_values(ascending=False)
    most_cor = most_cor[most_cor >= minimum_cor]

    columns_tox = []
    for i in range(len(most_cor.index)):
        columns_tox += [most_cor.index[i][0], most_cor.index[i][1]]
    columns_tox = sorted(list(set(columns_tox)))

    corr_mtx_toxicity = data.corr().loc[columns_tox, columns_tox]
    return corr_mtx_toxicity

def get_corr_with_target(data, target):
    df = data.copy()
    au_corr = df.corr().abs().unstack()
    labels_to_drop = set()
    cols = df.columns
    for i in range(0, df.shape[1]):
        for j in range(0, i+1):
            labels_to_drop.add((cols[i], cols[j]))
    most_cor = au_corr.drop(labels=labels_to_drop).sort_values(ascending=False)

    columns_tox = []
    for i in range(len(most_cor.index)):
        columns_tox += [most_cor.index[i][0], most_cor.index[i][1]]
    columns_tox = sorted(list(set(columns_tox)))

    corr_mtx_toxicity = t_data.corr().loc[columns_tox, columns_tox]
    return corr_mtx_toxicity
    