import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import config as cfg
import seaborn as sns
import ds_functions as ds

from sklearn.model_selection import train_test_split, StratifiedKFold, LeaveOneOut
import sklearn.metrics as metrics
from sklearn.tree import DecisionTreeClassifier

from sklearn.tree import export_graphviz
from subprocess import call
from strategies import KFold

from data import balance

def dtree(data: pd.DataFrame, target: str, balancing=None):
    df = data.copy()
    columns = df.columns
    
    y: np.ndarray = df.pop(target).values
    X: np.ndarray = df.values
    labels = pd.unique(y)

    trnY, prd_trn, tstY, prd_tst, trnX, tstX = KFold(X, y, 5)
    
    if (balancing != None):
        df = pd.DataFrame(data=np.hstack((trnX, np.array([trnY]).T)), columns=columns)
        df = balance(df, balancing, target)
        trnY = df.pop(target).to_numpy().T
        trnX = df.to_numpy()

    min_impurity_decrease = [0.025, 0.01, 0.005, 0.0025, 0.001]
    max_depths = [2, 5, 10, 15, 20, 25]
    criteria = ['entropy', 'gini']
    best = ('',  0, 0.0)
    last_best = 0
    best_tree = None

    plt.figure()
    fig, axs = plt.subplots(1, 2, figsize=(16, 4), squeeze=False)
    for k in range(len(criteria)):
        f = criteria[k]
        values = {}
        for d in max_depths:
            yvalues = []
            for imp in min_impurity_decrease:
                tree = DecisionTreeClassifier(max_depth=d, criterion=f, min_impurity_decrease=imp)
                tree.fit(trnX, trnY)
                prdY = tree.predict(tstX)
                cur_prd_trn = tree.predict(trnX)
                yvalues.append(metrics.accuracy_score(tstY, prdY))

                if yvalues[-1] > last_best:
                    best = (f, d, imp)
                    last_best = yvalues[-1]
                    best_tree = tree
                    prd_trn = cur_prd_trn

            values[d] = yvalues
        ds.multiple_line_chart(min_impurity_decrease, values, ax=axs[0, k], title='Decision Trees with %s criteria'%f,
                               xlabel='min_impurity_decrease', ylabel='accuracy', percentage=True)

    plt.show()
    print('Best results achieved with %s criteria, depth=%d and min_impurity_decrease=%1.3f ==> accuracy=%1.3f'%(best[0], best[1], best[2], last_best))
    
    ds.plot_evaluation_results(pd.unique(y), trnY, prd_trn, tstY, prd_tst)
    return best_tree
    
def dtree_graph(best_tree):
    dot_data = export_graphviz(best_tree, out_file='dtree.dot', filled=True, rounded=True, special_characters=True)
    # Convert to png
    call(['dot', '-Tpng', 'dtree.dot', '-o', 'dtree.png', '-Gdpi=600'])

    plt.figure(figsize = (14, 18))
    plt.imshow(plt.imread('dtree.png'))
    plt.axis('off')
    plt.show()