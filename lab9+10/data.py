import pandas as pd

# Covid deaths data
covid_data = pd.read_csv('covid19_pt.csv')
covid_data_raw = covid_data.copy()

# All deaths data
all_data = pd.read_csv('deaths_pt.csv')
all_data_raw = all_data.copy()

def get_covid_data():
    data = covid_data.copy()
    
    return data

def get_all_data():
    data = all_data.copy()
    
    return data