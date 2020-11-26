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

def showConfusionMatrix(trnX, tstX, y, trnY, tstY, best_tree):
    prd_trn = best_tree.predict(trnX)
    prd_tst = best_tree.predict(tstX)
    ds.plot_evaluation_results(pd.unique(y), trnY, prd_trn, tstY, prd_tst)
    
def naiveBayes(data, target, kfold=True, seed=None, balancing=None):
    data_nb = data.copy()
    
    y: np.ndarray = data_nb.pop(target).values
    X: np.ndarray = data_nb.values
    labels = pd.unique(y)
    
    if kfold:
        trnY, prd_trn, tstY, prd_tst, trnX, tstX = KFold(X, y, 5, seed)
    else:
        trnX, tstX, trnY, tstY = train_test_split(X, y, train_size=0.7, stratify=y, random_state=seed)
        prd_trn, prd_tst = None, None
        
    if (balancing != None):
        df = pd.DataFrame(data=np.hstack((trnX, np.array([trnY]).T)), columns=data.columns)
        df = balance(df, balancing, target)
        trnY = df.pop(target).to_numpy().T
        trnX = df.to_numpy()

    clf = GaussianNB()
    clf.fit(trnX, trnY)
    prd_trn = clf.predict(trnX)
    prd_tst = clf.predict(tstX)
    ds.plot_evaluation_results(pd.unique(y), trnY, prd_trn, tstY, prd_tst)
    
    estimators = {'GaussianNB': GaussianNB(),
              'MultinomialNB': MultinomialNB(),
              'BernoulliNB': BernoulliNB()}

    xvalues = []
    yvalues = []
    for clf in estimators:
        xvalues.append(clf)
        estimators[clf].fit(trnX, trnY)
        prdY = estimators[clf].predict(tstX)
        yvalues.append(metrics.accuracy_score(tstY, prdY))

    plt.figure()
    ds.bar_chart(xvalues, yvalues, title='Comparison of Naive Bayes Models', ylabel='accuracy', percentage=True)
    plt.show()

def KNN(data, target, kfold=True, quick=False, seed=None, balancing=None):
    data_knn = data.copy()
    
    y: np.ndarray = data_knn.pop(target).values
    X: np.ndarray = data_knn.values
    labels = pd.unique(y)
    
    if kfold:
        trnY, prd_trn, tstY, prd_tst, trnX, tstX = KFold(X, y, 5, seed)
    else:
        trnX, tstX, trnY, tstY = train_test_split(X, y, train_size=0.7, stratify=y, random_state=seed)
        prd_trn, prd_tst = None, None
        
    if (balancing != None):
        df = pd.DataFrame(data=np.hstack((trnX, np.array([trnY]).T)), columns=data.columns)
        df = balance(df, balancing, target)
        trnY = df.pop(target).to_numpy().T
        trnX = df.to_numpy()
        
    nvalues = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]
    dist = ['manhattan', 'euclidean', 'chebyshev']
    if (quick):
        nvalues = [1, 3, 5, 7, 9, 11]
    values = {}
    best = (0, '', None, 0)
    last_best = 0
    pbar = tqdm(total=(len(nvalues)*len(dist)))
    for d in dist:
        yvalues = []
        for n in nvalues:
            knn = KNeighborsClassifier(n_neighbors=n, metric=d)
            knn.fit(trnX, trnY)
            prdY = knn.predict(tstX)
            pbar.update(1)
            yvalues.append(metrics.accuracy_score(tstY, prdY))
            if yvalues[-1] > last_best:
                best = (n, d, knn, yvalues[-1])
                last_best = yvalues[-1]
        values[d] = yvalues

    pbar.close()
    plt.figure()
    ds.multiple_line_chart(nvalues, values, title='KNN variants', xlabel='n', ylabel='accuracy', percentage=True)
    plt.show()
    print('Best results with %d neighbors and %s, with accuracy %.2f'%(best[0], best[1], best[3]))
    
    prd_trn = best[2].predict(trnX)
    prd_tst = best[2].predict(tstX)
    ds.plot_evaluation_results(pd.unique(y), trnY, prd_trn, tstY, prd_tst)
    return (trnX, tstX, y, trnY, tstY, best[2])

def LogRegression(data, target, kfold=True, quick=False, seed=None, balancing=None):
    data_regression = data.copy()
    
    y: np.ndarray = data_regression.pop(target).values
    X: np.ndarray = data_regression.values
    labels = pd.unique(y)

    if kfold:
        trnY, prd_trn, tstY, prd_tst, trnX, tstX = KFold(X, y, 5, seed)
    else:
        trnX, tstX, trnY, tstY = train_test_split(X, y, train_size=0.7, stratify=y, random_state=seed)
        prd_trn, prd_tst = None, None
    
    if (balancing != None):
        df = pd.DataFrame(data=np.hstack((trnX, np.array([trnY]).T)), columns=data.columns)
        df = balance(df, balancing, target)
        trnY = df.pop(target).to_numpy().T
        trnX = df.to_numpy()
        
    n_iters = [50, 100, 150, 200, 250, 300]
    penalties = ["l2", "none"]
    if (quick):
        n_iters = [50, 100, 150, 200]
    best = (0, '')
    last_best = 0
    best_tree = None

    plt.figure()
    fig, axs = plt.subplots(1, 1, figsize=(ds.HEIGHT, ds.HEIGHT), squeeze=False)
    values = {}
    pbar = tqdm(total=(len(n_iters)*len(penalties)))
    for k in range(len(penalties)):
        p = penalties[k]
        yvalues = []
        for d in n_iters:
            lr = LogisticRegression(penalty=p, max_iter=d)
            lr.fit(trnX, trnY)
            prdY = lr.predict(tstX)
            pbar.update(1)
            yvalues.append(metrics.accuracy_score(tstY, prdY))
            if yvalues[-1] > last_best:
                best = (d, p)
                last_best = yvalues[-1]
                best_tree = lr

        values[p] = yvalues

    ds.multiple_line_chart(n_iters, values, ax=axs[0, 0], title='Logistic Regression',
                               xlabel='nr iterations', ylabel='accuracy', percentage=True)

    pbar.close()
    plt.show()
    print('Best results with %d iterations and %s penalty, with accuracy=%1.4f'%(best[0], best[1], last_best))
    # Get confusion matrix
    showConfusionMatrix(trnX, tstX, y, trnY, tstY, best_tree)
    res = (trnX, tstX, y, trnY, tstY, best_tree)
    
    # Get overfitting
    plt.figure()
    fig, axs = plt.subplots(1, 1, figsize=(16, 4), squeeze=False)
    values = {'Train':[], 'Test':[]}
    pbar = tqdm(total=(len(n_iters)))
    for iters in n_iters:
        tree = LogisticRegression(penalty=best[1], max_iter=iters)
        tree.fit(trnX, trnY)
        prdY = tree.predict(tstX)
        prdYTrain = tree.predict(trnX)
        pbar.update(1)
        values['Train'].append(metrics.accuracy_score(trnY, prdYTrain))
        values['Test'].append(metrics.accuracy_score(tstY, prdY))

    pbar.close()
    ds.multiple_line_chart(n_iters, values, ax=axs[0, 0], title='Overfitting for Logistic Regression',
                               xlabel='iterations', ylabel='accuracy', percentage=True)
    return res
    
    
def decisionTrees(data, target, kfold=True, quick=False, seed=None, balancing=None):
    data_forests = data.copy()
    
    y: np.ndarray = data_forests.pop(target).values
    X: np.ndarray = data_forests.values
    labels = pd.unique(y)

    if kfold:
        trnY, prd_trn, tstY, prd_tst, trnX, tstX = KFold(X, y, 5, seed)
    else:
        trnX, tstX, trnY, tstY = train_test_split(X, y, train_size=0.7, stratify=y, random_state=seed)
        prd_trn, prd_tst = None, None
        
    if (balancing != None):
        df = pd.DataFrame(data=np.hstack((trnX, np.array([trnY]).T)), columns=data.columns)
        df = balance(df, balancing, target)
        trnY = df.pop(target).to_numpy().T
        trnX = df.to_numpy()

    min_impurity_decrease = [0.025, 0.01, 0.005, 0.0025, 0.001]
    max_depths = [2, 5, 10, 15, 20, 25]
    criteria = ['entropy', 'gini']
    if (quick):
        min_impurity_decrease = [0.025, 0.01, 0.005, 0.0025, 0.001]
        max_depths = [2, 5, 10]
    best = ('',  0, 0.0)
    last_best = 0
    best_tree = None

    plt.figure()
    fig, axs = plt.subplots(1, 2, figsize=(16, 4), squeeze=False)
    pbar = tqdm(total=(len(min_impurity_decrease)*len(max_depths)*len(criteria)))
    for k in range(len(criteria)):
        f = criteria[k]
        values = {}
        for d in max_depths:
            yvalues = []
            for imp in min_impurity_decrease:
                tree = DecisionTreeClassifier(max_depth=d, criterion=f, min_impurity_decrease=imp)
                tree.fit(trnX, trnY)
                prdY = tree.predict(tstX)
                pbar.update(1)
                yvalues.append(metrics.accuracy_score(tstY, prdY))

                if yvalues[-1] > last_best:
                    best = (f, d, imp)
                    last_best = yvalues[-1]
                    best_tree = tree

            values[d] = yvalues
        ds.multiple_line_chart(min_impurity_decrease, values, ax=axs[0, k], title='Decision Trees with %s criteria'%f,
                               xlabel='min_impurity_decrease', ylabel='accuracy', percentage=True)

    pbar.close()
    plt.show()
    print('Best results achieved with %s criteria, depth=%d and min_impurity_decrease=%1.3f ==> accuracy=%1.3f'%(best[0], best[1], best[2], last_best))
    # Get confusion matrix
    showConfusionMatrix(trnX, tstX, y, trnY, tstY, best_tree)
    res = (trnX, tstX, y, trnY, tstY, best_tree)
    
    # Get overfitting
    plt.figure()
    fig, axs = plt.subplots(1, 1, figsize=(16, 4), squeeze=False)
    values = {'Train':[], 'Test':[]}
    pbar = tqdm(total=(len(min_impurity_decrease)))
    for imp in min_impurity_decrease:
        tree = DecisionTreeClassifier(max_depth=best[1], criterion=best[0], min_impurity_decrease=imp)
        tree.fit(trnX, trnY)
        prdY = tree.predict(tstX)
        prdYTrain = tree.predict(trnX)
        pbar.update(1)
        values['Train'].append(metrics.accuracy_score(trnY, prdYTrain))
        values['Test'].append(metrics.accuracy_score(tstY, prdY))

    pbar.close()
    ds.multiple_line_chart(min_impurity_decrease, values, ax=axs[0, 0], title='Overfitting for Decision Trees',
                               xlabel='min_impurity_decrease', ylabel='accuracy', percentage=True)
    return res

def randomForests(data, target, kfold=True, quick=False, seed=None, balancing=None):
    data_forests = data.copy()

    y: np.ndarray = data_forests.pop(target).values
    X: np.ndarray = data_forests.values
    labels = pd.unique(y)
    
    if kfold:
        trnY, prd_trn, tstY, prd_tst, trnX, tstX = KFold(X, y, 5, seed)
    else:
        trnX, tstX, trnY, tstY = train_test_split(X, y, train_size=0.7, stratify=y, random_state=seed)
        prd_trn, prd_tst = None, None
        
    if (balancing != None):
        df = pd.DataFrame(data=np.hstack((trnX, np.array([trnY]).T)), columns=data.columns)
        df = balance(df, balancing, target)
        trnY = df.pop(target).to_numpy().T
        trnX = df.to_numpy()

    n_estimators = [5, 10, 25, 50, 75, 100, 150, 200, 250, 300]
    max_depths = [5, 10, 25]
    max_features = [.1, .3, .5, .7, .9, 1]
    if (quick):
        n_estimators = [5, 10, 25, 50]
        max_depths = [5]
    best = ('', 0, 0)
    last_best = 0
    best_tree = None

    cols = len(max_depths)
    plt.figure()
    fig, axs = plt.subplots(1, cols, figsize=(cols*ds.HEIGHT, ds.HEIGHT), squeeze=False)
    pbar = tqdm(total=(len(n_estimators)*len(max_depths)*len(max_features)))
    for k in range(len(max_depths)):
        d = max_depths[k]
        values = {}
        for f in max_features:
            yvalues = []
            for n in n_estimators:
                rf = RandomForestClassifier(n_estimators=n, max_depth=d, max_features=f)
                rf.fit(trnX, trnY)
                prdY = rf.predict(tstX)
                pbar.update(1)
                yvalues.append(metrics.accuracy_score(tstY, prdY))
                if yvalues[-1] > last_best:
                    best = (d, f, n)
                    last_best = yvalues[-1]
                    best_tree = rf

            values[f] = yvalues
        ds.multiple_line_chart(n_estimators, values, ax=axs[0, k], title='Random Forests with max_depth=%d'%d,
                               xlabel='nr estimators', ylabel='accuracy', percentage=True)

    pbar.close()
    plt.show()
    print('Best results with depth=%d, %1.2f features and %d estimators, with accuracy=%1.4f'%(best[0], best[1], best[2], last_best))
    showConfusionMatrix(trnX, tstX, y, trnY, tstY, best_tree)
    return (trnX, tstX, y, trnY, tstY, best_tree)

def gradientBoosting(data, target, kfold=True, quick=False, seed=None, balancing=None):
    data_gradient = data.copy()
    y: np.ndarray = data_gradient.pop(target).values
    X: np.ndarray = data_gradient.values
    labels = pd.unique(y)

    if kfold:
        trnY, prd_trn, tstY, prd_tst, trnX, tstX = KFold(X, y, 5, seed)
    else:
        trnX, tstX, trnY, tstY = train_test_split(X, y, train_size=0.7, stratify=y, random_state=seed)
        prd_trn, prd_tst = None, None
        
    if (balancing != None):
        df = pd.DataFrame(data=np.hstack((trnX, np.array([trnY]).T)), columns=data.columns)
        df = balance(df, balancing, target)
        trnY = df.pop(target).to_numpy().T
        trnX = df.to_numpy()

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
                gb = GradientBoostingClassifier(n_estimators=n, max_depth=d, learning_rate=lr)
                gb.fit(trnX, trnY)
                prdY = gb.predict(tstX)
                pbar.update(1)
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
    showConfusionMatrix(trnX, tstX, y, trnY, tstY, best_tree)
    return (trnX, tstX, y, trnY, tstY, best_tree)