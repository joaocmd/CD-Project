import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import config as cfg
import seaborn as sns
import ds_functions as ds
import sklearn.metrics as metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, LeaveOneOut
from sklearn.ensemble import GradientBoostingClassifier
from imblearn.over_sampling import SMOTE
from tqdm import tqdm

def getBalancing(data, target):
    target_count = data[target].value_counts()
    plt.figure()
    plt.title('Class balance')
    plt.bar(target_count.index, target_count.values)
    plt.show()

    min_class = target_count.idxmin()
    ind_min_class = target_count.index.get_loc(min_class)

    print('Minority class:', target_count[ind_min_class])
    print('Majority class:', target_count[1-ind_min_class])
    print('Proportion:', round(target_count[ind_min_class] / target_count[1-ind_min_class], 2), ': 1')
    
    RANDOM_STATE = 42
    values = {'Original': [target_count.values[ind_min_class], target_count.values[1-ind_min_class]]}
    unbal = data.copy()

    df_class_min = unbal[unbal[target] == min_class]
    df_class_max = unbal[unbal[target] != min_class]

    df_under = df_class_max.sample(len(df_class_min))
    values['UnderSample'] = [target_count.values[ind_min_class], len(df_under)]

    df_over = df_class_min.sample(len(df_class_max), replace=True)
    values['OverSample'] = [len(df_over), target_count.values[1-ind_min_class]]

    smote = SMOTE(sampling_strategy='minority', random_state=RANDOM_STATE)
    y = unbal.pop(target).values
    X = unbal.values
    smote_X, smote_y = smote.fit_sample(X, y)
    smote_target_count = pd.Series(smote_y).value_counts()
    values['SMOTE'] = [smote_target_count.values[ind_min_class], smote_target_count.values[1-ind_min_class]]

    smote = pd.concat([pd.DataFrame(smote_X, columns=unbal.columns), pd.DataFrame(smote_y, columns=[target])], axis=1)

    fig = plt.figure()
    ds.multiple_bar_chart([target_count.index[ind_min_class], target_count.index[1-ind_min_class]], values,
                          title='Target', xlabel='frequency', ylabel='Class balance')
    plt.show()
    

def KFold(X, y, nfolds, seed=None):
    skf = StratifiedKFold(n_splits=nfolds, random_state=seed)

    opts=()
    accuracy = 0
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        min_impurity_decrease = [0.025, 0.01, 0.005, 0.0025, 0.001]
        max_depths = [2, 5, 10, 15, 20, 25]
        criteria = ['entropy', 'gini']
        
        for k in range(len(criteria)):
            f = criteria[k]
            for d in max_depths:
                for imp in min_impurity_decrease:
                    tree = DecisionTreeClassifier(max_depth=d, criterion=f, min_impurity_decrease=imp)
                    tree.fit(X_train, y_train)
                    prd_trn = tree.predict(X_train)
                    prd_tst = tree.predict(X_test)

                    curr_accuracy = metrics.accuracy_score(y_test, prd_tst)

                    if curr_accuracy > accuracy:
                        opts = (y_train, prd_trn, y_test, prd_tst, X_train, X_test)
                        accuracy = curr_accuracy
    return opts

def gradientBoosting(data, target, kfold=True, quick=False, seed=None):
    data_gradient = data.copy()
    y: np.ndarray = data_gradient.pop(target).values
    X: np.ndarray = data_gradient.values
    labels = pd.unique(y)

    if kfold:
        trnY, prd_trn, tstY, prd_tst, trnX, tstX = KFold(X, y, 5, seed)
    else:
        trnX, tstX, trnY, tstY = train_test_split(X, y, train_size=0.7, stratify=y)

    n_estimators = [5, 10, 25, 50, 75, 100, 150, 200, 250, 300]
    max_depths = [5, 10, 25]
    learning_rate = [.1, .3, .5, .7, .9]
    if (quick):
        n_estimators = [5, 10, 25, 50]
        max_depths = [5]
    best = ('', 0, 0)
    last_best = 0
    best_tree = None

    cols = len(max_depths)
    plt.figure()
    fig, axs = plt.subplots(1, cols, figsize=(cols*ds.HEIGHT, ds.HEIGHT), squeeze=False)
    pbar = tqdm(total=(len(n_estimators)*len(max_depths)*len(learning_rate)))
    for k in range(len(max_depths)):
        d = max_depths[k]
        values = {}
        for lr in learning_rate:
            yvalues = []
            for n in n_estimators:
                pbar.update(1)
                gb = GradientBoostingClassifier(n_estimators=n, max_depth=d, learning_rate=lr)
                gb.fit(trnX, trnY)
                prdY = gb.predict(tstX)
                yvalues.append(metrics.accuracy_score(tstY, prdY))
                if yvalues[-1] > last_best:
                    best = (d, lr, n)
                    last_best = yvalues[-1]
                    best_tree = gb
            values[lr] = yvalues
        ds.multiple_line_chart(n_estimators, values, ax=axs[0, k], title='Gradient Boosting with max_depth=%d'%d,
                               xlabel='nr estimators', ylabel='accuracy', percentage=True)

    pbar.close()
    plt.show()
    print('Best results with depth=%d, learning rate=%1.2f and %d estimators, with accuracy=%1.2f'%(best[0], best[1], best[2], last_best))
    return (trnX, tstX, y, trnY, tstY, best_tree)