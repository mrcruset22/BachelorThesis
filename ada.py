import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, chi2, RFE, SelectFromModel, SequentialFeatureSelector
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV

def univariateDesirability(table, proportion):
    desirability = round(len(table.keys())/2)
    for i in table.keys():
        if '_id' in i:
            pass
        else:
            try:
                pd.to_numeric(table[i])
                desirability += 2
            except ValueError:
                if table.groupby(i).count().iloc[:, 0].sort_values()[-1:][0]/len(table) < proportion\
                        and table[i].nunique() <= 10:
                    desirability += 1
    return desirability

def multivariateDesirability(table, cols, proportion):
    clf = LinearDiscriminantAnalysis()
    keys = table.keys()

    num = []
    out = []

    for i in keys:
        if '_id' in i:
            pass
        else:
            try:
                pd.to_numeric(table[i])
                num.append(i)
            except ValueError:
                if table.groupby(i).count().iloc[:, 0].sort_values()[-1:][0] / len(table) < proportion \
                        and table[i].nunique() <= 10:
                    out.append(i)
    table_lda = table[list(num)]
    table_lda = table_lda.apply(pd.to_numeric, errors='coerce')
    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    imp.fit(table_lda)
    table_lda = imp.transform(table_lda)
    table_lda = pd.DataFrame(table_lda, columns=num)
    try:
        table_lda["lda"] = clf.fit_transform(table_lda.drop(columns=["class"]), table_lda["class"])
        corr = abs(table_lda.drop(columns=["class"]).corr()["lda"][:-1])
        corr = corr.sort_values(ascending=False)
        out = []
        for i in corr[corr.gt(0.01)].keys():
            if i in cols:
                out.append(i)
    except ValueError:
        pass

    return out

def ada(base, tables, connections, proportion):
    current_id = connections["from_key"][0]
    for i,j in zip(tables, connections.iterrows()):
        uni_desirability = univariateDesirability(i, proportion)
        if uni_desirability > 5:
            i.set_index(j[1][3], drop=True, inplace=True)
            if current_id != j[1][3]:
                try:
                    base.set_index(j[1][3], drop=True, inplace=True)
                except KeyError:
                    pass
                    # print("This cannot be joineeed")
                current_id = j[1][3]
            cols_to_use = i.columns.difference(base.columns)
            to_join = pd.merge(base.drop(base.tail(round(len(base)*0.1)).index), i[cols_to_use],
                               left_index=True, right_index=True, how='left')
            columns_to_join = multivariateDesirability(to_join, [k for k in cols_to_use if '_id' not in k], proportion)
            if len(columns_to_join) > 0:
                columns_to_join.extend([k for k in cols_to_use if '_id' in k])
                base = pd.merge(base, i[columns_to_join], left_index=True, right_index=True, how='left')
    base = base[base.columns.drop(list(base.filter(regex='_id')))]
    base = pd.get_dummies(base)
    base = base.apply(pd.to_numeric, errors='coerce')
    return base