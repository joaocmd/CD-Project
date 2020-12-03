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
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, BaggingClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm
from data import balance
import statistics
from ds_functions import *

def KFold(data, target, try_params, balancing=None):
    df = data.copy()
    y: np.ndarray = df.pop(target).values
    X: np.ndarray = df.values

    skf = StratifiedKFold(n_splits=5, random_state=42069)

    opts=()
    accuracy = 0
    scores = []
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        if (balancing != None):
            df = pd.DataFrame(data=np.hstack((X_train, np.array([y_train]).T)), columns=data.columns)
            df = balance(df, balancing, target)
            y_train = df.pop(target).to_numpy().T
            X_train = df.to_numpy()

        scores += [try_params(X_train, X_test, y_train, y_test, pd.unique(y))]
    
    avg, std = calculate_avg_results(scores)
    return (avg, std)

def calculate_avg_results(results):
    metrics = ['Accuracy', 'Recall', 'Specificity', 'Precision']
    res = {}
    std = {}
    for metric in metrics:
        for params in results[0]:
            if params not in res:
                res[params] = {}
                std[params] = {}
            res[params][metric] = [statistics.mean([fold[params][metric][0] for fold in results]),
                                   statistics.mean([fold[params][metric][1] for fold in results])]
            std[params][metric] = [np.std([fold[params][metric][0] for fold in results]),
                                  np.std([fold[params][metric][1] for fold in results])]
    return (res, std)

def DecisionTreeTryParams(X_train, X_test, y_train, y_test, labels):
    min_impurity_decrease = [0.025, 0.01, 0.005, 0.0025, 0.001]
    max_depths = [2, 5, 10, 15, 20, 25]
    criteria = ['entropy', 'gini']

    results = {}
    pbar = tqdm(total=(len(min_impurity_decrease)*len(max_depths)*len(criteria)))
    for k in range(len(criteria)):
        f = criteria[k]
        for d in max_depths:
            for imp in min_impurity_decrease:
                tree = DecisionTreeClassifier(max_depth=d, criterion=f, min_impurity_decrease=imp)
                tree.fit(X_train, y_train)
                prd_trn = tree.predict(X_train)
                prd_tst = tree.predict(X_test)
                pbar.update(1)

                results[(f, d, imp)] = ds.calc_evaluations_results(labels, y_train, prd_trn, y_test, prd_tst)

    pbar.close()
    return results

def DecisionTreesKFold(data, target, balancing=None):
    results, error = KFold(data, target, DecisionTreeTryParams, balancing)
    best = [params for params in results if
            all(results[params]['Accuracy'][1] >= results[x]['Accuracy'][1] for x in results)][0]

    fig, axs = plt.subplots(1, 2, figsize=(2 * 4, 4))
    print('criteria, max_depth, min_impurity_decrease')
    multiple_bar_chart(['Train', 'Test'], results[best], error=error[best], ax=axs[0], title="Model's performance over Train and Test sets")
    return (best, results[best])

def kNNTryParams(X_train, X_test, y_train, y_test, labels):
    nvalues = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]
    dist = ['manhattan', 'euclidean', 'chebyshev']
    results = {}
    pbar = tqdm(total=(len(nvalues)*len(dist)))
    for d in dist:
        for n in nvalues:
            knn = KNeighborsClassifier(n_neighbors=n, metric=d)
            knn.fit(X_train, y_train)
            prd_tst = knn.predict(X_test)
            prd_trn = knn.predict(X_train)
            pbar.update(1)

            results[(d, n)] = ds.calc_evaluations_results(labels, y_train, prd_trn, y_test, prd_tst)

    pbar.close()
    return results

def kNNKFold(data, target, balancing=None):
    results, std = KFold(data, target, kNNTryParams, balancing)
    best = [params for params in results if
            all(results[params]['Accuracy'][1] >= results[x]['Accuracy'][1] for x in results)][0]

    fig, axs = plt.subplots(1, 2, figsize=(2 * 4, 4))
    print('dist, n')
    multiple_bar_chart(['Train', 'Test'], results[best], error=std[best], ax=axs[0], title="Model's performance over Train and Test sets")
    return (best, results[best])

def NaiveBayesTryParams(X_train, X_test, y_train, y_test, labels):
    estimators = {'GaussianNB': GaussianNB(),
                  'MultinomialNB': MultinomialNB(),
                  'BernoulliNB': BernoulliNB()}

    results = {}
    pbar = tqdm(total=(len(estimators)))
    for clf in estimators:
        estimators[clf].fit(X_train, y_train)
        prd_tst = estimators[clf].predict(X_test)
        prd_trn = estimators[clf].predict(X_train)
        pbar.update(1)
        
        results[clf] = ds.calc_evaluations_results(labels, y_train, prd_trn, y_test, prd_tst)
    pbar.close()
    
    return results

def NaiveBayesKFold(data, target, balancing=None):
    results, error = KFold(data, target, NaiveBayesTryParams, balancing)
    best = [params for params in results if
            all(results[params]['Accuracy'][1] >= results[x]['Accuracy'][1] for x in results)][0]

    fig, axs = plt.subplots(1, 2, figsize=(2 * 4, 4))
    multiple_bar_chart(['Train', 'Test'], results[best], error=error[best], ax=axs[0], title="Model's performance over Train and Test sets")
    return (best, results[best])

def RandomForestsTryParams(X_train, X_test, y_train, y_test, labels):
    n_estimators = [5, 10, 25, 50, 75, 100, 150, 200, 250, 300]
    max_depths = [5, 10, 25]
    max_features = [.1, .3, .5, .7, .9, 1]

    results = {}
    pbar = tqdm(total=(len(n_estimators)*len(max_depths)*len(max_features)))
    for k in range(len(max_depths)):
        d = max_depths[k]
        values = {}
        for f in max_features:
            yvalues = []
            for n in n_estimators:
                rf = RandomForestClassifier(n_estimators=n, max_depth=d, max_features=f)
                rf.fit(X_train, y_train)
                prd_trn = rf.predict(X_train)
                prd_tst = rf.predict(X_test)
                pbar.update(1)

                results[(max_depths[k], f, n)] = ds.calc_evaluations_results(labels, y_train, prd_trn, y_test, prd_tst)

    pbar.close()
    return results

def RandomForestsKFold(data, target, balancing=None):
    results, error = KFold(data, target, RandomForestsTryParams, balancing)
    best = [params for params in results if
            all(results[params]['Accuracy'][1] >= results[x]['Accuracy'][1] for x in results)][0]

    fig, axs = plt.subplots(1, 2, figsize=(2 * 4, 4))
    print('max_depth, max_features, n_estimators')
    multiple_bar_chart(['Train', 'Test'], results[best], error=error[best], ax=axs[0], title="Model's performance over Train and Test sets")
    return (best, results[best])

def XGBoostTryParams(X_train, X_test, y_train, y_test, labels):
    n_estimators = [5, 10, 25, 50, 75, 100, 150, 200, 250, 300]
    max_depths = [5, 10, 25]
    learning_rate = [.1, .3, .5, .7, .9]

    results = {}
    pbar = tqdm(total=(len(n_estimators)*len(max_depths)*len(learning_rate)))
    for d in max_depths:
        values = {}
        for lr in learning_rate:
            yvalues = []
            for n in n_estimators:
                gb = GradientBoostingClassifier(n_estimators=n, max_depth=d, learning_rate=lr)
                gb.fit(X_train, y_train)
                prd_trn = gb.predict(X_train)
                prd_tst = gb.predict(X_test)
                pbar.update(1)

                results[(d, lr, n)] = ds.calc_evaluations_results(labels, y_train, prd_trn, y_test, prd_tst)

    pbar.close()
    return results

def XGBoostKFold(data, target, balancing=None):
    results, error = KFold(data, target, XGBoostTryParams, balancing)
    best = [params for params in results if
            all(results[params]['Accuracy'][1] >= results[x]['Accuracy'][1] for x in results)][0]

    fig, axs = plt.subplots(1, 2, figsize=(2 * 4, 4))
    print('max_depth, learning_rate, n_estimators')
    multiple_bar_chart(['Train', 'Test'], results[best], error=error[best], ax=axs[0], title="Model's performance over Train and Test sets")
    return (best, results[best])