#!/usr/bin/env python
# coding: utf-8

# In[1]:
import argparse
import pandas as pd
import json
import functools as ft
from sklearn import svm
from pandas import ExcelWriter
from pandas import ExcelFile
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
#from sklearn.neighbors.nca import NeighborhoodComponentsAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import TSNE

from sklearn import model_selection
from sklearn.model_selection import KFold
import xgboost as xgb
import random
# from imblearn.over_sampling import SMOTE
from collections import Counter
from mpl_toolkits.mplot3d import Axes3D
from time import time
from sklearn import manifold
from matplotlib.ticker import NullFormatter
import operator, itertools
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import warnings
warnings.filterwarnings('ignore')
#get_ipython().run_line_magic('matplotlib', 'inline')
# Harsh: Use Python[pytorch] kernel <others, please ignore this line>


# In[2]:


# ### Considering on dummy data
# # Input data :
# # X = num_edges x num_methods,
# # y = predictions (I - 0/1 values; II - 0/1/2/3 hop connections)

# E = 10000 # number of edges
# M = 6 # number of methods
# X = np.random.rand(E, M)
# y = np.random.choice([0, 1], size=E, p=[.9, .1]) # percentage labels


# ### Loading the train data: Yeast

# ## Visualizing the data
# Sometimes it can be helpful, so why not!

# In[7]:


def visual2D(X, color):
    Axes3D
    n_points = len(color)
    S = 30 # point size for figures

    n_neighbors = 10
    n_components = 2

    # fig = plt.figure(figsize=(15, 8))
    fig = plt.figure(figsize=(20, 8))
    plt.suptitle("2D projection with %i points, %i neighbors"
            % (n_points, n_neighbors), fontsize=14)

    methods = ['standard', 'ltsa', 'hessian', 'modified']
    labels = ['LLE', 'LTSA', 'Hessian LLE', 'Modified LLE']

    for i, method in enumerate(methods):
        t0 = time()
        Y = manifold.LocallyLinearEmbedding(n_neighbors, n_components,
                eigen_solver='dense',
                method=method).fit_transform(X)
        t1 = time()
        print("%s: %.2g sec" % (methods[i], t1 - t0))

        ax = fig.add_subplot(252 + i)
        plt.scatter(Y[:, 0], Y[:, 1], c=color, cmap=plt.cm.Spectral, s=S)
        plt.title("%s (%.2g sec)" % (labels[i], t1 - t0))
        ax.xaxis.set_major_formatter(NullFormatter())
        ax.yaxis.set_major_formatter(NullFormatter())
        plt.axis('tight')

    t0 = time()
    Y = manifold.Isomap(n_neighbors, n_components).fit_transform(X)
    t1 = time()
    print("Isomap: %.2g sec" % (t1 - t0))
    ax = fig.add_subplot(257)
    plt.scatter(Y[:, 0], Y[:, 1], c=color, cmap=plt.cm.Spectral, s=S)
    plt.title("Isomap (%.2g sec)" % (t1 - t0))
    ax.xaxis.set_major_formatter(NullFormatter())
    ax.yaxis.set_major_formatter(NullFormatter())
    plt.axis('tight')


    t0 = time()
    mds = manifold.MDS(n_components, max_iter=100, n_init=1)
    Y = mds.fit_transform(X)
    t1 = time()
    print("MDS: %.2g sec" % (t1 - t0))
    ax = fig.add_subplot(258)
    plt.scatter(Y[:, 0], Y[:, 1], c=color, cmap=plt.cm.Spectral, s=S)
    plt.title("MDS (%.2g sec)" % (t1 - t0))
    ax.xaxis.set_major_formatter(NullFormatter())
    ax.yaxis.set_major_formatter(NullFormatter())
    plt.axis('tight')


    t0 = time()
    se = manifold.SpectralEmbedding(n_components=n_components,
            n_neighbors=n_neighbors)
    Y = se.fit_transform(X)
    t1 = time()
    print("SpectralEmbedding: %.2g sec" % (t1 - t0))
    ax = fig.add_subplot(259)
    plt.scatter(Y[:, 0], Y[:, 1], c=color, cmap=plt.cm.Spectral, s=S)
    plt.title("SpectralEmbedding (%.2g sec)" % (t1 - t0))
    ax.xaxis.set_major_formatter(NullFormatter())
    ax.yaxis.set_major_formatter(NullFormatter())
    plt.axis('tight')

    t0 = time()
    tsne = manifold.TSNE(n_components=n_components, init='pca', random_state=5)
    Y = tsne.fit_transform(X)
    t1 = time()
    print("t-SNE: %.2g sec" % (t1 - t0))
    ax = fig.add_subplot(2, 5, 10)
    plt.scatter(Y[:, 0], Y[:, 1], c=color, cmap=plt.cm.Spectral, s=S)
    plt.title("t-SNE (%.2g sec)" % (t1 - t0))
    ax.xaxis.set_major_formatter(NullFormatter())
    ax.yaxis.set_major_formatter(NullFormatter())
    plt.axis('tight')

    plt.show()


clf_VERSION = 'v1'
K = 5 # 5 fold cv

def plot_confusion_matrix(cm, classes,
        normalize=False,
        title='Confusion matrix',
        cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.figure(figsize = (5,3))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = np.array(cm)
        cm = np.around(cm/cm.sum(axis=1)[:, None]*100).astype('int')
        print("Percentage confusion matrix")
        print(cm.sum(axis=1))
    else:
        print('Confusion matrix, without normalization')

#    print(cm)
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black")
        plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return


def show_confusion_matrix(y, pred_array):
    y = np.array(y).astype(int)
    y_pred = np.array(pred_array)

    cnf_matrix = confusion_matrix(y, y_pred)
    np.set_printoptions(precision=2)
    sorted_cnf_matrix = cnf_matrix
    class_names = ['yes', 'no'] #[0, 1]

    plot_confusion_matrix(sorted_cnf_matrix, classes=class_names,
            title='Confusion matrix, without normalization')
    plt.show()
    return


def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def get_auc_plot(y, scores):
    y = np.array(y).astype(int)
    fpr, tpr, thresholds = metrics.roc_curve(y, scores)
#    print(fpr, tpr, thresholds)
    # Getting accuracy, sensitivity and accuracy plot for varying thresholds.
    accuracy_array = []
    sensitivity_array = [x*100 for x in tpr]
    specificity_array = [(1-x)*100 for x in fpr]#(1-fpr)*100
    for i, th in enumerate(thresholds):
        pred_array = []
        for s in scores:
            if s>th:
                pred_array.append(1)
            else:
                pred_array.append(0)
        accuracy_array.append(accuracy_score(y, pred_array))
#         print('For threshold : ', th, '->', ' Accuracy ', accuracy_array[i],
#               ' Sensitivity ', sensitivity_array[i], 'specificity ', specificity_array[i])
#         show_confusion_matrix(y, pred_array)
#     print('accuracy_array', accuracy_array)
#     print('sensitivity_array', sensitivity_array)
#     print('specificity_array', specificity_array)
#     print('thresholds_array', thresholds)

    roc_auc = metrics.auc(fpr, tpr)
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([-0.001, 1])
    plt.ylim([0, 1.001])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
    for i in range(len(fpr)):
        if fpr[i] > 0.01:
            break
    aupr = metrics.average_precision_score(y, scores)
    return roc_auc, aupr, tpr[i], fpr[i]

def get_auc(y, scores):
    y = np.array(y).astype(int)
    fpr, tpr, thresholds = metrics.roc_curve(y, scores)
    roc_auc = metrics.auc(fpr, tpr)

#     for i in range(len(fpr)):
#         if fpr[i] > 0.01:
#             break
    aupr = metrics.average_precision_score(y, scores)
    return [roc_auc, aupr]#, tpr[i], fpr[i]


def trainXGB(Xtrain, ytrain, Xtest, ytest, colFeats, output_image=None, scale_pos_weight=None):
    dtrain = xgb.DMatrix(Xtrain,label=ytrain)
    dtest = xgb.DMatrix(Xtest,label=ytest)
    #print('Setting XGB params')
    evallist  = [(dtest,'test'), (dtrain,'train')]

    param = {}
    # use softmax multi-class classification
#    param['objective'] = 'multi:softprob'#'multi:softmax'
    param['objective'] = 'binary:logistic' #'multi:softprob'
    # scale weight of positive examples
    param['eta'] = 0.01
    param['max_depth'] = 7
    param['gamma'] = 0
#    param['silent'] = 1
    param['nthread'] = 6
#    param['subsample'] = 0.5#0.7 # number of examples for 1 tree (subsampled from total)
#    param['colsample_bytree'] = 0.5#0.7 # ratio of columns for training 1 tree
#    param['num_class'] = NUM_CLASS
    param['eval_metric'] = 'auc'#'mlogloss' #auc

    # CLASS Imbalance handling!
#    param['scale_pos_weight'] = 10#190/10 # sum(negative cases) / sum(positive cases)
    if scale_pos_weight is not None:
        param['scale_pos_weight'] = scale_pos_weight

#    param['booster'] = 'gblinear' #'dart' #'gblinear' # default is tree booster
#    param['lambda'] = 1
#    param['alpha'] = 1

    num_round = 220#60
    #print('training the XGB classifier')
    bst = xgb.train(param, dtrain, num_round, evallist, early_stopping_rounds=100, verbose_eval=False)
    #print('training completed, printing the relative importance: \
            #      (feature id: importance value)')
    importance = bst.get_fscore()
    importance = sorted(importance.items(), key=operator.itemgetter(1))

    #print(importance)

    # we will print from df1 dataframe, getting the corresponding feature names.
    df1 = pd.DataFrame(importance, columns=['feature', 'fscore'])
    # Normalizing the feature scores
    df1['fscore'] = df1['fscore'] / df1['fscore'].sum()

    #print(df1)
    # adding a column of feature name
    # DEFINE as global
    #colFeats = [c for c in df_train.columns if c not in ['prediction', 'edge']]
    #colFeats = ['clr', 'aracne', 'grnboost', 'mrnet', 'tinge', 'wgcna']


    # column_names = df_train.columns[:-1]
    df1['feature_names'] = pd.Series([colFeats[int(f[0].replace("f", ""))] for f in importance])

    df1.plot()
    df1.plot(kind='barh', x='feature_names', y='fscore', legend=False, figsize=(6, 10))
    plt.title('XGBoost Feature Importance')
    plt.xlabel('relative importance')
    if output_image is None:
        plt.gcf().savefig('feature_importance_xgb.png')
    else:
        plt.gcf().savefig(output_image)
    #plt.show()
#    print 'Saving the models'
#    bst.save_model(name+'_xgb_v'+clf_VERSION+'.model')
#    bst.dump_model(name+'_xgb_v'+clf_VERSION+'_dump.raw.txt')
#    bst.dump_model(name+'_xgb_v'+clf_VERSION+'_dump.raw.txt',name+'_xgb_v'+clf_VERSION+'_featmap.txt')
    return bst


def classifier_train(X, y, runXGB, Xtest, ytest, colFeats,
        output_image=None, scale_pos_weight=None, pca_comp = 0): # pca = 0 means no PCA applied
    #print('Normalising the input data...')
    #print(X)
    #print(y)
    scaler = StandardScaler()#MinMaxScaler()#StandardScaler()
    scaler.fit(X)
    scaledX = scaler.transform(X)
    if pca_comp != 0:
        pca = PCA(n_components = pca_comp)
        pca.fit(scaledX)
        pca_scaledX = pca.transform(scaledX)
    else:
        pca_scaledX = scaledX
        pca =0

    if runXGB == 1:
        #print('Running the XGB classifier')
        clf = trainXGB(pca_scaledX, y, scaler.transform(Xtest), ytest,
                       colFeats, output_image, scale_pos_weight)
        index = 1
    return scaler, pca, clf, index

def classifier_prediction(X, clf, index, scaler, pca):
    scaledX = scaler.transform(X)
    pca_scaledX = scaledX
    if pca != 0:
        pca_scaledX = pca.transform(scaledX)
    if index==1:# XGB
        pca_scaledXG = xgb.DMatrix(pca_scaledX)
        pred_array = clf.predict(pca_scaledXG, ntree_limit=clf.best_iteration)
    return pred_array

def classifier_test(X, y, clf, index, scaler, pca):
    scaledX = scaler.transform(X)
    pca_scaledX = scaledX
    if pca != 0:
        pca_scaledX = pca.transform(scaledX)
    if index==1:# XGB
        pca_scaledXG = xgb.DMatrix(pca_scaledX, label=y)
        pred_array = clf.predict(pca_scaledXG, ntree_limit=clf.best_iteration)
        #.reshape(y.shape[0], NUM_CLASS)#, ntree_limit=clf.best_iteration)
        scores = pred_array
    auc = get_auc_plot(y, scores)
    # Compute confusion matrix
    #show_confusion_matrix(y, pred_array)
    return pred_array, auc# error


def runRankAvg1(data, methods):
    # get the ranks for the individual methods
    df_edges = pd.DataFrame(data['edge'])
    for i, m in enumerate(methods):
        rankM = pd.concat([data['edge'], data[m]], axis=1)
        # sort according to the method value
        rankM.sort_values(by=[m], inplace=True, ascending=False)
        rankM.reset_index(drop=True, inplace=True)
        rankNum = pd.Series([i for i in range(0, rankM.shape[0])], name='rank_m'+str(i))
        rankM = pd.concat([rankM, rankNum], axis=1)
        df_edges = pd.merge(left=df_edges, right=rankM, left_on='edge', right_on='edge')
        del df_edges[m]

    df_edges['rankAvg'] = df_edges.mean(axis=1)
    mx_rank = df_edges['rankAvg']
    df_edges['rankAvg'] = 0 - df_edges['rankAvg']
    # reverse sorting the avgRank as auc needs scores.
    #return get_auc(data['prediction'], df_edges['rankAvg'][::-1])
    return get_auc(data['prediction'], df_edges['rankAvg'])


def runRankAvg2(data, methods):
    # get the ranks for the individual methods
    df_edges = data[methods + ["prediction"]]
    df_edges['rankAvg'] = 1/df_edges[methods].rank(ascending=False).mean(axis=1)
    # reverse sorting the avgRank as auc needs scores.
    return get_auc(data['prediction'], df_edges['rankAvg'])


def runRankAvg3(data, methods):
    # get the ranks for the individual methods
    df_edges = data[methods + ["prediction"]]
    df_edges['rankAvg'] = df_edges[methods].rank(ascending=False).mean(axis=1)
    # reverse sorting the avgRank as auc needs scores.
    x, y = get_auc(1 - data['prediction'], df_edges['rankAvg'])
    return [x, y]

def scalesum_stats(adf, methods):
    acx = adf[methods]
    astd = acx.std(axis=0)
    amean = acx.mean(axis=0)
    sx1 = (adf[methods] - amean)/astd
    adf['wt'] =sx1.sum(axis=1)
    aupr = metrics.average_precision_score(adf.prediction, adf.wt)
    auroc = metrics.roc_auc_score(adf.prediction, adf.wt)
    return auroc, aupr

def scalelsum_stats(df, colFeats):
    adf = pd.concat([allOnes, allZeros])
    adf[['s', 't']] = adf['edge'].str.split('-', expand=True)
    ax1 = adf[methods+['s']].rename(columns={'s':'node'})
    ax2 = adf[methods+['t']].rename(columns={'t':'node'})
    acx = pd.concat([ax1,ax2])
    astd = acx.groupby("node").std()
    amean = acx.groupby("node").mean()
    amean.reset_index()
    astd.reset_index()
    tx1 = pd.DataFrame(adf['s']).rename(columns={'s':'node'}) 
    tx2 = pd.DataFrame(adf['t']).rename(columns={'t':'node'})
    smean = tx1.merge(amean, on="node")  
    sstd = tx1.merge(astd, on="node")   
    tmean = tx2.merge(amean, on="node") 
    tstd = tx2.merge(astd, on="node")
    e1 = (adf[methods] - smean[methods])/sstd[methods]
    e2 = (adf[methods] - tmean[methods])/tstd[methods]
    ex1 = e1 **2
    ex2 = e2 **2
    sx1 = np.sqrt(ex1 + ex2)
    adf['wt'] =sx1.sum(axis=1)
    aupr = metrics.average_precision_score(adf.prediction, adf.wt)
    auroc = metrics.roc_auc_score(adf.prediction, adf.wt)
    return auroc, aupr


def runScaleLSum(allOnes, allZeros, methods):
    adf = pd.concat([allOnes, allZeros])
    adf.fillna(0)
    adf[['s', 't']] = adf['edge'].str.split('-', expand=True)
    ax1 = adf[methods+['s']].rename(columns={'s':'node'})
    ax2 = adf[methods+['t']].rename(columns={'t':'node'})
    acx = pd.concat([ax1,ax2])
    astd = acx.groupby("node").std()
    amean = acx.groupby("node").mean()
    amean.reset_index()
    astd.reset_index()
    tx1 = pd.DataFrame(adf['s']).rename(columns={'s':'node'}) 
    tx2 = pd.DataFrame(adf['t']).rename(columns={'t':'node'})
    smean = tx1.merge(amean, on="node")  
    sstd = tx1.merge(astd, on="node")   
    tmean = tx2.merge(amean, on="node") 
    tstd = tx2.merge(astd, on="node")
    e1 = (adf[methods] - smean[methods])/sstd[methods]
    e2 = (adf[methods] - tmean[methods])/tstd[methods]
    ex1 = e1 **2
    ex2 = e2 **2
    sx1 = np.sqrt(ex1 + ex2)
    adf['scaleLSum'] =sx1.sum(axis=1)
    adf['scaleLSum'].replace(np.inf, 0, inplace=True)
    adf['scaleLSum'].replace(-np.inf, 0, inplace=True)
    print(adf, adf['prediction'].isna().sum(),np.isinf(adf['scaleLSum']).sum())
    return get_auc(adf['prediction'], adf['scaleLSum'])


def runScaleSum(allOnes, allZeros, methods):
    adf = pd.concat([allOnes, allZeros])
    acx = adf[methods]
    astd = acx.std(axis=0)
    amean = acx.mean(axis=0)
    sx1 = (adf[methods] - amean)/astd
    adf['wt'] =sx1.sum(axis=1)
    adf['scaleSum'] =sx1.sum(axis=1)
    return get_auc(adf['prediction'], adf['scaleSum'])



def individual_method(data, methods, rankAvg=False, printIt=True):
    #methods = [c for c in data.columns if c not in ['prediction', 'edge']]
    results = {}
    for m in methods:
        auc = get_auc(data['prediction'], data[m])
        results[m] = auc

    # Adding the rank Avg method
    if rankAvg:
        results['RankAvg-Sorted'] = runRankAvg1(data, methods)
        results['RankAvg-Reciprocal'] = runRankAvg2(data, methods)
        results['RankAvg-UpDown'] = runRankAvg3(data, methods)
        results['RankAvg-ScaleSum'] = scalesum_stats(data, methods)
    #print(results)
    if printIt is True:
        for r in results:
            print(r, results[r])
    return results

def pandas_classifier(df_train, df_test, runXGB, colFeats,
        output_image=None, scale_pos_weight=None, K=10, rAvg=False,
        fold_auc=True):
    print('Performing ' + str(K) + '-fold cross validation with ', colFeats)
    auc_fold = []
    pr_fold = []
    #colFeats = [c for c in df_train.columns if c not in ['prediction', 'edge']]

    for k in range(K):# performing K fold validation
        #if k == 0: # running only for k'th fold
            # print('Fold_num = ' + str(k))
            #train_rows = [i for i in range(len(df_train)) if i%K!=k]
            datatrain = df_train.loc[[i for i in range(len(df_train)) if i%K!=k]] # training
            #valid_rows = [i for i in range(len(df_train)) if i%K==k]
            datavalid = df_train.loc[[i for i in range(len(df_train)) if i%K==k]] # taking every k'th example
#             Xtrain =  #.iloc[:, 0:-1]
#             ytrain =  #.iloc[:, -1]
#             Xvalid =  #.iloc[:, 0:-1]
#             yvalid = #.iloc[:, -1]
            #print('--------------------------------------------------------------')
            #print('Calling the classifier to train')
            #print(datatrain[colFeats].head())
            #print(datavalid[colFeats].head())
            scaler, pca, clf, index = classifier_train(datatrain[colFeats], datatrain['prediction'],
                    runXGB, datavalid[colFeats], datavalid['prediction'],
                    colFeats, output_image, scale_pos_weight)
            #print('Analysing the test predictions for fold num ', k)
            pred_array, auc = classifier_test(datavalid[colFeats],
                    datavalid['prediction'], clf, index, scaler, 0)
            auc_fold.append(auc[0])
            pr_fold.append(auc[1])
            print('test auc = '+str(auc[0]) )
            if fold_auc is True:
                individual_method(datavalid, colFeats, rankAvg=rAvg)
            #print('------------------------------------------------------------')
    avg_auc = sum(np.array(auc_fold))/int(K) if K > 0 else 0.0
    avg_aupr = sum(np.array(pr_fold))/int(K) if K > 0 else 0.0
    std_auc = np.std(np.array(auc_fold)) if K > 0 else 0.0
    std_aupr = np.std(np.array(pr_fold)) if K > 0 else 0.0
    if K != 0:
        print('************************************************************************')
        #print(auc_fold)#, sum(np.array(auc_fold))/int(K))
        #print(pr_fold)
        print('AVG ', str(K), 'AUROC', str(avg_auc), '+/-', str(std_auc))
        print('AVG ', str(K), 'AUPR', str(avg_aupr), '+/-', str(std_aupr))
        print('************************************************************************')

#     pred_array, auc = classifier_test(df_test[colFeats], df_test['prediction'], clf, index, scaler, 0)
#     print('TEST AUC on standalone data = ', auc[0])
#     print('individual methods: ', individual_method(df_test))
    return [clf, index, scaler, avg_auc, avg_aupr]

def load_v3_dataset():
    # In[6]:
    df = pd.read_csv('data/yeast-edge-weights-v3.csv', sep=',')
    # In[7]:
    # dropping pcc as not present for test arabidopsis
    del df['pcc']
    # del df['grnboost']
    # In[8]:
    # for c in df.columns:
    #     print(c, df[c].value_counts())
    df['prediction'].value_counts()
    # In[9]:
    allOnes = df[df['prediction']==1]
    allZeros = df[df['prediction']==0]
    numOnes = len(allOnes)
    numZeros = len(allZeros)
    trainPTS = 20000 # train + valid
    testPTS = 200000 # test
    # choose 20K points for training
    # Randomly choose 200K points for testing
    # evenly divide the ones among train/test
    df_train = pd.concat([allOnes[:numOnes//2], allZeros[:trainPTS]], ignore_index=True)
    df_test = pd.concat([allOnes[numOnes//2:], allZeros[trainPTS: trainPTS+testPTS]], ignore_index=True)
    return(df_train, df_test)


def load_v4_dataset():
    # In[6]:
    df = pd.read_csv('data/yeast-edge-weights-v4.csv', sep=',')
    del df['pcc']
    # del df['grnboost']
    # for c in df.columns:
    #     print(c, df[c].value_counts())
    df['prediction'].value_counts()
    #
    allOnes = df[df['prediction']==1]
    allZeros = df[df['prediction']==0]
    numOnes = len(allOnes)
    numZeros = len(allZeros)
    trainPTS = 20000 # train + valid
    testPTS = 200000 # test
    # choose 20K points for training
    # Randomly choose 200K points for testing
    # evenly divide the ones among train/test
    df_train = pd.concat([allOnes[:numOnes//2], allZeros[:trainPTS]], ignore_index=True)
    df_test = pd.concat([allOnes[numOnes//2:], allZeros[trainPTS: trainPTS+testPTS]], ignore_index=True)
    return (df_train, df_test, allOnes, allZeros)

def load_v5_dataset():
    # In[6]:
    df = pd.read_csv('data/yeast-edge-weights-v5.csv', sep=',')
    del df['pcc']
    # del df['grnboost']
    # for c in df.columns:
    #     print(c, df[c].value_counts())
    df['prediction'].value_counts()
    #
    allOnes = df[df['prediction']==1]
    allZeros = df[df['prediction']==0]
    numOnes = len(allOnes)
    numZeros = len(allZeros)
    trainPTS = 20000 # train + valid
    testPTS = 200000 # test
    # choose 20K points for training
    # Randomly choose 200K points for testing
    # evenly divide the ones among train/test
    df_train = pd.concat([allOnes[:numOnes//2], allZeros[:trainPTS]], ignore_index=True)
    df_test = pd.concat([allOnes[numOnes//2:], allZeros[trainPTS: trainPTS+testPTS]], ignore_index=True)
    return (df_train, df_test, allOnes, allZeros)



def analysis_data_v3():
    df_train, df_test = load_v3_dataset()
    # In[10]:


    print(df_train.min(axis=0), df_train.max(axis=0))


    # In[8]:


    # Visualize the train data: 1500 points
    colFeats = [c for c in df_train.columns if c not in ['prediction', 'edge']]

    Xviz = np.array(df_train[colFeats])
    yviz = np.array(df_train['prediction'])
    vizPTS = np.random.choice(len(Xviz), 1500)
    #print(Xviz[:vizPTS, :].shape, len(yviz[:vizPTS]), sum(yviz[:vizPTS]))
    visual2D(Xviz[vizPTS, :], yviz[vizPTS])


def analysis_data_v4_xgboost():
    df_train, df_test, allOnes, allZeros = load_v4_dataset()
    print("Loaded v4 dataset ")
    print(df_train.head())
    print(df_test.head())
    df_train.fillna(0)
    df_test.fillna(0)
    output_image="v4-analysis.png"
    colFeats = [c for c in df_train.columns if c not in ['prediction', 'edge']]
    trained_params = pandas_classifier(df_train, df_test, 1, colFeats, output_image)
    print('******************** Indiv. methods on Test **********************')
    individual_method(df_test, colFeats, rankAvg=True)
    print('scaleLSum on Test :', runScaleLSum(allOnes, allZeros, colFeats))
    print('scaleSum  on Test :', runScaleSum(allOnes, allZeros, colFeats))
    print('******************************************************************')
    colFeats2 = ["clr", "grnboost", "aracne", "mrnet", "tinge", "wgcna"]
    trained_params2 = pandas_classifier(df_train, df_test, 1, colFeats2, output_image)
    print('******************* Indiv. methods on Top 6 **********************')
    individual_method(df_test, colFeats2, rankAvg=True)
    print('scaleLSum Top 6 :', runScaleLSum(allOnes, allZeros, colFeats2))
    print('scaleSum  Top 6 :', runScaleSum(allOnes, allZeros, colFeats2))
    print('******************************************************************')
    colFeats3 = ["clr", "grnboost", "aracne", "mrnet", "tinge", "wgcna", "genie3"]
    trained_params3 = pandas_classifier(df_train, df_test, 1, colFeats3, output_image)
    print('******************* Indiv. methods on Top 7 **********************')
    individual_method(df_test, colFeats3, rankAvg=True)
    print('scaleLSum Top 7 :', runScaleLSum(allOnes, allZeros, colFeats3))
    print('scaleSum  Top 7 :', runScaleSum(allOnes, allZeros, colFeats3))
    print('******************************************************************')

def analysis_data_v5_xgboost():
    df_train, df_test, allOnes, allZeros = load_v5_dataset()
    print("Loaded v4 dataset ")
    print(df_train.head())
    print(df_test.head())
    output_image="v4-analysis.png"
    colFeats = [c for c in df_train.columns if c not in ['prediction', 'edge']]
    trained_params = pandas_classifier(df_train, df_test, 1, colFeats, output_image)
    print('******************** Indiv. methods on Test **********************')
    individual_method(df_test, colFeats, rankAvg=True)
    print('scaleSum  on Test :', runScaleSum(allOnes, allZeros, colFeats))
    print('scaleLSum on Test :', runScaleLSum(allOnes, allZeros, colFeats))
    print('******************************************************************')
    colFeats2 = ["clr", "grnboost", "aracne", "mrnet", "tinge", "wgcna"]
    trained_params2 = pandas_classifier(df_train, df_test, 1, colFeats2, output_image)
    print('******************* Indiv. methods on Top 6 **********************')
    individual_method(df_test, colFeats2, rankAvg=True)
    print('scaleLSum Top 6 :', runScaleLSum(allOnes, allZeros, colFeats2))
    print('scaleSum  Top 6 :', runScaleSum(allOnes, allZeros, colFeats2))
    print('******************************************************************')
    colFeats3 = ["clr", "grnboost", "aracne", "mrnet", "tinge", "wgcna", "genie3"]
    trained_params3 = pandas_classifier(df_train, df_test, 1, colFeats3, output_image)
    print('******************* Indiv. methods on Top 7 **********************')
    individual_method(df_test, colFeats3, rankAvg=True)
    print('scaleLSum Top 7 :', runScaleLSum(allOnes, allZeros, colFeats3))
    print('scaleSum  Top 7 :', runScaleSum(allOnes, allZeros, colFeats3))
    print('******************************************************************')



def analysis_data_v3_xgboost():
    df_train, df_test = load_v3_dataset()

    colFeats = [c for c in df_train.columns if c not in ['prediction', 'edge']]
    trained_params = pandas_classifier(df_train, df_test, 1, colFeats)

    print('individual methods on test: ', individual_method(df_test, colFeats, rankAvg=True))


    abdOnes  = pd.read_csv('data/true_network_positive_edges.csv', sep=',')
    abdOnes['prediction'] = 1
    abdZeros = pd.read_csv('data/true_network_negative_edges.csv', sep=',')
    abdZeros['prediction'] = 0
    df_abd   = pd.concat([abdOnes, abdZeros], ignore_index=True)
    #del df_abd['grnboost']
    df_abd


    # In[14]:


    # # Imputation step with zeros
    df_abd = df_abd.fillna(0) # DOES NOT work that well


    # # Imputation with the min values of the big file Arabidopsis
    # colFeats = [c for c in df_abd.columns if c not in ['prediction', 'edge']]
    # for c in colFeats:
    #     df_abd[c] = df_abd[c].fillna(minAbd_vals[c])
    # print(minAbd_vals)

    # # Imputation with the min values of the train data in yeast
    # minYeast_vals = df_train.min(axis=0)
    # colFeats = [c for c in df_abd.columns if c not in ['prediction', 'edge']]
    # for c in colFeats:
    #     df_abd[c] = df_abd[c].fillna(minYeast_vals[c])


    # # Scaling the test data in the ranges similar to training of Yeast
    # minYeast = df_train.min(axis=0)
    # maxYeast = df_train.max(axis=0)
    # minAbd = df_abd.min(axis=0)
    # maxAbd = df_abd.max(axis=0)
    # colFeats = [c for c in df_abd.columns if c not in ['prediction', 'edge']]
    # for c in colFeats:
    #     df_abd[c] = df_abd[c].fillna(minYeast_vals[c])
    #     # (x-min)/(max-min)*(b-a) + a
    #     df_abd[c] = (df_abd[c] - df_abd[c].min())/(df_abd[c].max()-df_abd[c].min())*(maxYeast[c]-minYeast[c]) + minYeast[c]


    df_abd


    # In[15]:


    # predicting using the trained params
    clf, index, scaler, _, _ = trained_params

    print('Check col order: ', colFeats)
    pred_array, auc = classifier_test(df_abd[colFeats], df_abd['prediction'], clf, index, scaler, 0)
    print('ENSEMBLE AUC on Arabidopsis data = ', auc[0])
    print('individual methods on arabidopsis: ')
    individual_method(df_abd, colFeats)


    # ## RESULTS: imputation with zeros
    # ENSEMBLE AUC on Arabidopsis data =  0.7127507970060296
    # individual methods on arabidopsis:
    # {'aracne': 0.7068620486520202, 'tinge': 0.6973737785016287, 'wgcna': 0.601063829787234, 'grnboost': 0.7553862880310486, 'mrnet': 0.6480330410977893, 'clr': 0.5384988564696098}

    # ## RESULTS: Imputation with min value from the big arabidopsis file
    # ENSEMBLE AUC on Arabidopsis data =  0.641143703652367
    # individual methods on arabidopsis:
    # {'aracne': 0.7068620486520202, 'tinge': 0.6973737785016287, 'wgcna': 0.601063829787234, 'grnboost': 0.7553862880310486, 'mrnet': 0.6480330410977893, 'clr': 0.5384988564696098}

    # In[ ]:





    # In[ ]:





    # In[ ]:





    # ## Finding values for imputation
    # Less than the minimum value given in the big predictions file

    # In[22]:


    df_fullAbd = pd.read_csv('data/arabidopsis-edges-final-test-v1.csv', sep=',')


    # In[23]:


    df_fullAbd


    # In[25]:


    minAbd_vals = df_fullAbd.min(axis=0)


    # In[26]:


    minAbd_vals


    # In[34]:


    maxAbd_vals = df_fullAbd.max(axis=0)
    maxAbd_vals


    # In[ ]:


    # df_train yeast

    # edge          STA1-YBR112C
    # clr                 -2.787
    # aracne                   0
    # grnboost                 0
    # mrnet                    0
    # tinge           0.00180598
    # wgcna          2.37121e-39
    # dtype: object

    # edge          YPR065W-YPR201W
    # clr                   62.2546
    # aracne                0.98017
    # grnboost              1603.52
    # mrnet                0.998569
    # tinge                0.969508
    # wgcna                0.581019
    # dtype: object

    # In[62]:


    df_abd['clr'].min()



def athaliana_individual_network():
    df_train, df_test = load_v3_dataset()
    colFeats = [c for c in df_train.columns if c not in ['prediction', 'edge']]
    trained_params = pandas_classifier(df_train, df_test, 1, colFeats)
    ath_types = ['chemical', 'development', 'flower', 'hormone-aba-iaa-ga-br' ,'hormone-ja-sa-ethylene',
            'leaf', 'light', 'nutrients', 'root', 'rosette', 'seed', 'seedling1wk', 'seedling2wk',
            'shoot', 'stress-light', 'stress-other', 'stress-pathogen', 'stress-salt-drought',
            'stress-temperature', 'wholeplant']

    # predicting using the trained params
    clf, index, scaler, _, _ = trained_params

    for i, t in enumerate(ath_types):
        print('******************************************')
        print('ath type: ', t)
        athOnes  = pd.read_csv('data/athaliana_raw/'+ath_types[i]+'-positives.csv', sep=',')
        athOnes['prediction'] = 1
        athZeros = pd.read_csv('data/athaliana_raw/'+ath_types[i]+'-negatives.csv', sep=',')
        athZeros['prediction'] = 0
        df_ath   = pd.concat([athOnes, athZeros], ignore_index=True)
        #del df_abd['grnboost']
        # print(df_ath)
        # # Imputation step with zeros
        df_ath = df_ath.fillna(0)


        #print('Check col order: ', colFeats)
        pred_array, auc = classifier_test(df_ath[colFeats], df_ath['prediction'], clf, index, scaler, 0)
        print('ENSEMBLE AUC on Arabidopsis data = ', auc[0])
        print('individual methods on arabidopsis: ')
        individual_method(df_ath)


# ## Training and testing on the arabidopsis tissues and environments
# tissue_types = ['flower', 'leaf', 'rosette', 'seed', 'shoot']
# environment_types = []

# In[13]:

def union_net(df_tis, union_feats):
    data_dct = {'edge' : df_tis.edge}
    for x in union_feats:
        data_dct[x["name"]] = df_tis[x["feats"]].max(axis=1)
    data_dct['prediction'] = df_tis.prediction
    df = pd.DataFrame(data=data_dct)
    return df

def athaliana_unionavg_train(train_dir, union_feats):
    tisOnes  = pd.read_csv(train_dir+'/all-positives.csv', sep=',')
    tisOnes['prediction'] = 1
    print(tisOnes.shape)
    # Zeros
    tisZeros = pd.read_csv(train_dir+'/all-negatives.csv', sep=',')
    tisZeros['prediction'] = 0
    df_tis= pd.concat([tisOnes, tisZeros], ignore_index=True)
    df_tis = df_tis.fillna(0)
    df_union = union_net(df_tis, union_feats)
    df_union.wt = 1/df_union.rank(ascending=False).mean(axis=1)
    aupr = metrics.average_precision_score(df_union.prediction, df_union.wt)
    auroc = metrics.roc_auc_score(df_union.prediction, df_union.wt)
    print("FEATS", "NFEATS", "AUROC", "AUPR")
    print("Union-Max", sum(len(x) for x in union_feats), auroc, aupr)
    methods = [x["name"] for x in union_feats]
    auroc, aupr = scalesum_stats(df_union, methods)
    print("all-scalelsum", len(methods), auroc, aupr)
    for x in methods:
        aupr = metrics.average_precision_score(df_union.prediction, df_union[x])
        auroc = metrics.roc_auc_score(df_union.prediction, df_union[x])
        print(x, 20, auroc, aupr)

def avgrank_stats(df, colFeats):
    df_tis = df[colFeats + ["prediction"]]
    df_tis.wt = 1/df_tis[colFeats].rank(ascending=False).mean(axis=1)
    aupr = metrics.average_precision_score(df_tis.prediction, df_tis.wt)
    auroc = metrics.roc_auc_score(df_tis.prediction, df_tis.wt)
    return auroc, aupr




def athaliana_rankavg_train(train_dir, colFeats):
    tisOnes  = pd.read_csv(train_dir+'/all-positives.csv', sep=',')
    tisOnes['prediction'] = 1
    #print(tisOnes.shape)
    # Zeros
    tisZeros = pd.read_csv(train_dir+'/all-negatives.csv', sep=',')
    tisZeros['prediction'] = 0
    df_tis= pd.concat([tisOnes, tisZeros], ignore_index=True)
    df_tis = df_tis.fillna(0)
    print("FEATS", "NFEATS", "AUROC", "AUPR")
    auroc, aupr = avgrank_stats(df_tis, colFeats)
    print("all-avergrank", len(colFeats), auroc, aupr)
    auroc, aupr = scalesum_stats(df_tis, colFeats)
    print("all-scalelsum", len(colFeats), auroc, aupr)
    methods = ["aracne", "clr", "grnboost", "mrnet", "tinge", "wgcna"]
    labels = ["ARACNe-AP", "CLR", "GRNBoost", "MRNET", "TINGe", "WGCNA"]
    for x,z in zip(methods, labels):
        xfeats = [y for y in colFeats if y.endswith(x)]
        auroc, aupr = avgrank_stats(df_tis, xfeats)
        print(" & ".join(str(r) for r in [z+ " Rank Avg.", len(xfeats), round(auroc, 4), round(aupr, 4)]))
    for x,z in zip(methods, labels):
        xfeats = [y for y in colFeats if y.endswith(x)]
        auroc, aupr = scalesum_stats(df_tis, xfeats)
        print(" & ".join(str(r) for r in [z+ "\\textit{ScaleSum}", len(xfeats), round(auroc, 4), round(aupr, 4)]))
 

def athaliana_integ_train(train_dir, colFeats, output_image,
                         pos_ratio=0.5, neg_ratio=0.5, scale_pos_weight=None,
                         random_seed=5):
    # TODO: split half and half
    if random_seed is not None:
        random.seed(random_seed)
    # Ones
    tisOnes  = pd.read_csv(train_dir+'/all-positives.csv', sep=',')
    tisOnes['prediction'] = 1
    print(tisOnes.shape)
    nrow1 = int(tisOnes.shape[0])
    nselect1 = int(nrow1*pos_ratio)
    lselect1 = random.sample(list(range(nrow1)), nselect1)
    # Zeros
    tisZeros = pd.read_csv(train_dir+'/all-negatives.csv', sep=',')
    tisZeros['prediction'] = 0
    nrow2 = int(tisZeros.shape[0])
    nselect2 = int(nrow2*neg_ratio)
    lselect2 = random.sample(list(range(nrow2)), nselect2)
    # 
    df_tis= pd.concat([tisOnes.iloc[lselect1, ], tisZeros.iloc[lselect2,]], ignore_index=True)
    df_tis = df_tis.fillna(0)
    df_excl = pd.concat([tisOnes.drop(lselect1, axis=0), tisZeros.drop(lselect2, axis=0)], ignore_index=True)
    trained_params_tissue = pandas_classifier(df_tis, df_excl, 1, colFeats,
                                              output_image, scale_pos_weight)
    return trained_params_tissue, df_tis, df_excl

def athaliana_ensemble_train(train_dir, tissue_types_train,
                             colFeats, output_image, df_test = None, fold_auc=True):
    # training
    df_tis_train = pd.DataFrame([])
    for i, t in enumerate(tissue_types_train):
        #print(train_dir, i, t)
        tisOnes  = pd.read_csv(train_dir+'/'+t+'-positives.csv', sep=',')
        tisOnes['prediction'] = 1
        tisZeros = pd.read_csv(train_dir+'/'+t+'-negatives.csv', sep=',')
        tisZeros['prediction'] = 0
        df_tis= pd.concat([tisOnes, tisZeros], ignore_index=True)
        # # Imputation step with zeros
        df_tis = df_tis.fillna(0)
        df_tis_train = pd.concat([df_tis_train, df_tis], ignore_index=True)

    if df_test is None:
       trained_params_tissue = pandas_classifier(df_tis_train, df_tis_train, 1, colFeats, output_image, fold_auc=fold_auc)
    else:
       trained_params_tissue = pandas_classifier(df_tis_train, df_test, 1, colFeats, output_image, fold_auc=fold_auc)
    return trained_params_tissue


def athaliana_ensemble_aupr(tissue_types=None):
    train_dir = 'data/athaliana_raw/'
    default_types = ['flower', 'leaf','light','stress-temperature', 'hormone-ja-sa-ethylene']
    if tissue_types is None:
        tissue_types_train = default_types
    else:
        tissue_types_train = tissue_types

    colFeats = ['clr', 'aracne', 'grnboost', 'mrnet', 'tinge', 'wgcna']
    # testing
    # predicting using the trained params
    tissue_types_test = ['root', 'rosette', 'seed', 'seedling1wk', 'seedling2wk', 'shoot', 'wholeplant',
            'chemical', 'nutrients', 'stress-light', 'stress-other', 'stress-pathogen',
            'stress-salt-drought', 'development',
            'hormone-aba-iaa-ga-br']


    clf, index, scaler, _, _ = athaliana_ensemble_train(train_dir, tissue_types_train, colFeats)

    for i, t in enumerate(tissue_types_test):
        print('******************************************')
        print('tissue type: ', t)
        tisOnes  = pd.read_csv('data/athaliana_raw/'+t+'-positives.csv', sep=',')
        tisOnes['prediction'] = 1
        tisZeros = pd.read_csv('data/athaliana_raw/'+t+'-negatives.csv', sep=',')
        tisZeros['prediction'] = 0
        df_tis   = pd.concat([tisOnes, tisZeros], ignore_index=True)

        # # Imputation step with zeros
        df_tis = df_tis.fillna(0)


        #print('Check col order: ', colFeats)
        pred_array, auc = classifier_test(df_tis[colFeats], df_tis['prediction'], clf, index, scaler, 0)
        print('ENSEMBLE Arabidopsis data AUC = ', auc[0], ' AUPR = ', auc[1])
        print('individual methods on arabidopsis: [auroc, aupr]')
        ind_results = individual_method(df_tis, colFeats, rankAvg=False)
        ensemble_compare = True
        for r in ind_results:
            #print(r, ind_results[r]<auc[0])
            ensemble_compare &= ind_results[r][0]<auc[0]
        print('\n ENSEMBLE better? ', ensemble_compare, '\n')


def athaliana_ensemble_predict1(network_file, output_file, tissue_types=None):
    train_dir = 'data/athaliana_raw/'
    default_types = ['flower', 'leaf','light','stress-temperature', 'hormone-ja-sa-ethylene']
    if tissue_types is None:
        tissue_types_train = default_types
    else:
        tissue_types_train = tissue_types

    colFeats = ['clr', 'aracne', 'grnboost', 'mrnet', 'tinge', 'wgcna']
    # testing
    # predicting using the trained params
    tissue_types_test = ['root', 'rosette', 'seed', 'seedling1wk',
            'seedling2wk', 'shoot', 'wholeplant', 'chemical', 'nutrients',
            'stress-light', 'stress-other', 'stress-pathogen',
            'stress-salt-drought', 'development', 'hormone-aba-iaa-ga-br']


    output_image = output_file.split(".")[0]  + ".png"
    print("Features Image :", output_image)
    clf, index, scaler, _, _ = athaliana_ensemble_train(train_dir, tissue_types_train, colFeats, output_image)

    df_tis  = pd.read_csv(network_file, sep=',')
    df_tis = df_tis.fillna(0)
    print("Loaded Network file : ", network_file)
    pred_array = classifier_prediction(df_tis[colFeats], clf, index, scaler, 0)
    print("Finished prediction for Network file : ", network_file)
    df_tis['wt'] = pred_array
    if output_file is not None:
        df_tis.to_csv(output_file, sep="\t", index=False)

def athaliana_ensemble_predict2(network_file, output_file, tissue_types=None):
    train_dir = 'data/athaliana_raw2/'
    default_types = ['flower', 'leaf','light','stress-temperature', 'hormone-ja-sa-ethylene']
    if tissue_types is None:
        tissue_types_train = default_types
    else:
        tissue_types_train = tissue_types

    colFeats = ['clr', 'aracne', 'grnboost', 'mrnet', 'tinge', 'wgcna']
    # testing
    # predicting using the trained params
    tissue_types_test = ['root', 'rosette', 'seed', 'seedling1wk',
            'seedling2wk', 'shoot', 'wholeplant', 'chemical', 'nutrients',
            'stress-light', 'stress-other', 'stress-pathogen',
            'stress-salt-drought', 'development', 'hormone-aba-iaa-ga-br']


    output_image = output_file.split(".")[0]  + ".png"
    print("Features Image :", output_image)
    clf, index, scaler, _, _ = athaliana_ensemble_train(train_dir, tissue_types_train, colFeats, output_image)

    df_tis  = pd.read_csv(network_file, sep=',')
    df_tis = df_tis.fillna(0)
    print("Loaded Network file : ", network_file)
    pred_array = classifier_prediction(df_tis[colFeats], clf, index, scaler, 0)
    print("Finished prediction for Network file : ", network_file)
    df_tis['wt'] = pred_array
    if output_file is not None:
        df_tis.to_csv(output_file, sep="\t", index=False)


def athaliana_ensemble_predict3(network_file, output_file, tissue_types=None):
    train_dir = 'data/athaliana_raw/'
    default_types = ['flower', 'leaf','light','stress-temperature', 'hormone-ja-sa-ethylene']
    if tissue_types is None:
        tissue_types_train = default_types
    else:
        tissue_types_train = tissue_types

    #colFeats = ['clr', 'aracne', 'grnboost', 'mrnet', 'tinge', 'wgcna']
    colFeats = ['clr', 'aracne', 'grnboost', 'mrnet', 'tinge']
    # testing
    # predicting using the trained params
    tissue_types_test = ['root', 'rosette', 'seed', 'seedling1wk',
            'seedling2wk', 'shoot', 'wholeplant', 'chemical', 'nutrients',
            'stress-light', 'stress-other', 'stress-pathogen',
            'stress-salt-drought', 'development', 'hormone-aba-iaa-ga-br']


    output_image = output_file.split(".")[0]  + ".png"
    print("Features Image :", output_image)
    clf, index, scaler, _, _ = athaliana_ensemble_train(train_dir, tissue_types_train, colFeats, output_image)

    df_tis  = pd.read_csv(network_file, sep=',')
    df_tis = df_tis.fillna(0)
    print("Loaded Network file : ", network_file)
    pred_array = classifier_prediction(df_tis[colFeats], clf, index, scaler, 0)
    print("Finished prediction for Network file : ", network_file)
    df_tis['wt'] = pred_array
    if output_file is not None:
        df_tis.to_csv(output_file, sep="\t", index=False)


def athaliana_ensemble_predict4(network_file, output_file, tissue_types=None):
    train_dir = 'data/athaliana_raw/'
    default_types = ['flower', 'leaf','light','stress-temperature', 'hormone-ja-sa-ethylene']
    if tissue_types is None:
        tissue_types_train = default_types
    else:
        tissue_types_train = tissue_types

    #colFeats = ['clr', 'aracne', 'grnboost', 'mrnet', 'tinge', 'wgcna']
    colFeats = ['clr', 'aracne', 'grnboost', 'tinge']
    # testing
    # predicting using the trained params
    tissue_types_test = ['root', 'rosette', 'seed', 'seedling1wk',
            'seedling2wk', 'shoot', 'wholeplant', 'chemical', 'nutrients',
            'stress-light', 'stress-other', 'stress-pathogen',
            'stress-salt-drought', 'development', 'hormone-aba-iaa-ga-br']


    output_image = output_file.split(".")[0]  + ".png"
    print("Features Image :", output_image)
    clf, index, scaler, _, _ = athaliana_ensemble_train(train_dir, tissue_types_train, colFeats, output_image)

    df_tis  = pd.read_csv(network_file, sep=',')
    df_tis = df_tis.fillna(0)
    print("Loaded Network file : ", network_file)
    pred_array = classifier_prediction(df_tis[colFeats], clf, index, scaler, 0)
    print("Finished prediction for Network file : ", network_file)
    df_tis['wt'] = pred_array
    if output_file is not None:
        df_tis.to_csv(output_file, sep="\t", index=False)


def athaliana_ensemble_predict5a(network_file, output_file, tissue_types=None):
    train_dir = 'data/athaliana_raw/'
    default_types = ['flower', 'leaf','light','stress-temperature', 'hormone-ja-sa-ethylene']
    if tissue_types is None:
        tissue_types_train = default_types
    else:
        tissue_types_train = tissue_types

    #colFeats = ['clr', 'aracne', 'grnboost', 'mrnet', 'tinge', 'wgcna']
    colFeats = ['clr', 'grnboost', 'tinge']
    # testing
    # predicting using the trained params
    tissue_types_test = ['root', 'rosette', 'seed', 'seedling1wk',
            'seedling2wk', 'shoot', 'wholeplant', 'chemical', 'nutrients',
            'stress-light', 'stress-other', 'stress-pathogen',
            'stress-salt-drought', 'development', 'hormone-aba-iaa-ga-br']


    output_image = output_file.split(".")[0]  + ".png"
    print("Features Image :", output_image)
    clf, index, scaler, _, _ = athaliana_ensemble_train(train_dir, tissue_types_train, colFeats, output_image)

    df_tis  = pd.read_csv(network_file, sep=',')
    df_tis = df_tis.fillna(0)
    print("Loaded Network file : ", network_file)
    pred_array = classifier_prediction(df_tis[colFeats], clf, index, scaler, 0)
    print("Finished prediction for Network file : ", network_file)
    df_tis['wt'] = pred_array
    if output_file is not None:
        df_tis.to_csv(output_file, sep="\t", index=False)

def athaliana_ensemble_predict5b(network_file, output_file, tissue_types=None):
    train_dir = 'data/athaliana_raw/'
    default_types = ['flower', 'leaf','light','stress-temperature', 'hormone-ja-sa-ethylene']
    if tissue_types is None:
        tissue_types_train = default_types
    else:
        tissue_types_train = tissue_types

    #colFeats = ['clr', 'aracne', 'grnboost', 'mrnet', 'tinge', 'wgcna']
    colFeats = ['clr', 'aracne', 'grnboost']
    # testing
    # predicting using the trained params
    tissue_types_test = ['root', 'rosette', 'seed', 'seedling1wk',
            'seedling2wk', 'shoot', 'wholeplant', 'chemical', 'nutrients',
            'stress-light', 'stress-other', 'stress-pathogen',
            'stress-salt-drought', 'development', 'hormone-aba-iaa-ga-br']


    output_image = output_file.split(".")[0]  + ".png"
    print("Features Image :", output_image)
    clf, index, scaler, _, _ = athaliana_ensemble_train(train_dir, tissue_types_train, colFeats, output_image)

    df_tis  = pd.read_csv(network_file, sep=',')
    df_tis = df_tis.fillna(0)
    print("Loaded Network file : ", network_file)
    pred_array = classifier_prediction(df_tis[colFeats], clf, index, scaler, 0)
    print("Finished prediction for Network file : ", network_file)
    df_tis['wt'] = pred_array
    if output_file is not None:
        df_tis.to_csv(output_file, sep="\t", index=False)

def athaliana_ensemble_predict6(network_file, output_file, tissue_types=None):
    train_dir = 'data/athaliana_raw/'
    default_types = ['flower', 'leaf','light','stress-temperature', 'hormone-ja-sa-ethylene']
    if tissue_types is None:
        tissue_types_train = default_types
    else:
        tissue_types_train = tissue_types

    #colFeats = ['clr', 'aracne', 'grnboost', 'mrnet', 'tinge', 'wgcna']
    colFeats = ['clr', 'grnboost']
    # testing
    # predicting using the trained params
    tissue_types_test = ['root', 'rosette', 'seed', 'seedling1wk',
            'seedling2wk', 'shoot', 'wholeplant', 'chemical', 'nutrients',
            'stress-light', 'stress-other', 'stress-pathogen',
            'stress-salt-drought', 'development', 'hormone-aba-iaa-ga-br']

    output_image = output_file.split(".")[0]  + ".png"
    print("Features Image :", output_image)
    clf, index, scaler, _, _ = athaliana_ensemble_train(train_dir, tissue_types_train, colFeats, output_image)

    df_tis  = pd.read_csv(network_file, sep=',')
    df_tis = df_tis.fillna(0)
    print("Loaded Network file : ", network_file)
    pred_array = classifier_prediction(df_tis[colFeats], clf, index, scaler, 0)
    print("Finished prediction for Network file : ", network_file)
    df_tis['wt'] = pred_array
    if output_file is not None:
        df_tis.to_csv(output_file, sep="\t", index=False)


def athaliana_ensemble_predict7(output_file):
    train_dir = 'data/athaliana_raw_filtered/'
    all_types = ['flower', 'leaf', 'root', 'rosette', 'seed', 'seedling1wk',
                 'seedling2wk', 'shoot', 'wholeplant', 
                 'chemical', 'nutrients','light','stress-temperature',
                 'stress-light', 'stress-other', 'stress-pathogen',
                 'stress-salt-drought', 'development', 
                 'hormone-aba-iaa-ga-br', 'hormone-ja-sa-ethylene']
    all_tissue_types = ['flower', 'leaf', 'root', 'rosette', 'seed', 'seedling1wk',
                 'seedling2wk', 'shoot', 'wholeplant']

    types1 = ['flower']
    types2 = ['leaf']
    types3 = ['seedling1wk']
    types4 = ['flower', 'leaf']
    types5 = ['flower', 'root']
    types6 = ['flower', 'seedling1wk']

    colFeats = ['wgcna', 'mrnet', 'clr', 'grnboost', 'tinge', 'aracne']
    #colFeats = ['clr', 'grnboost']
    # testing
    # predicting using the trained params


    test_dir = "data/athaliana_raw_filtered/"
    all_pos_dfs = {x:pd.read_csv(test_dir+x+"-positives.csv") for x in all_tissue_types}
    all_neg_dfs = {x:pd.read_csv(test_dir+x+"-negatives.csv") for x in all_tissue_types}
    ##
    ##
    all_lst = []
    for tissue_types_train in [types1, types2, types3, types4, types5, types6]:
        output_image = output_file.split(".")[0]  + ".png"
        print("Output Image : ", output_image)
        clf, index, scaler, _, _ = athaliana_ensemble_train(train_dir, tissue_types_train, colFeats, output_image)
        tissue_types_test = list(set(all_tissue_types) - set(tissue_types_train))
        ensemble_dct = { x: 'R' for x in tissue_types_train}
        for tx in tissue_types_test:
            pos_df = all_pos_dfs[tx]
            print(pos_df)
            pos_df.fillna(0, inplace=True)
            pos_df['prediction'] = 1
            neg_df = all_neg_dfs[tx]
            neg_df.fillna(0, inplace=True)
            neg_df['prediction'] = 0
            df_test   = pd.concat([pos_df, neg_df], ignore_index=True)
            # 
            print("Tissue : ", tx, len(df_test.columns), len(colFeats), df_test.shape, df_test['prediction'].isna().sum())
            pred_array, auc = classifier_test(df_test[colFeats], df_test['prediction'],
                                              clf, index, scaler, 0)
            print('ENSEMBLE Arabidopsis data AUC = ', auc[0], ' AUPR = ', auc[1])
            print('individual methods on arabidopsis: [auroc, aupr]')
            ind_results = individual_method(df_test, colFeats, rankAvg=False, printIt=False)
            ensemble_compare = True
            for r in ind_results:
                #print(r, ind_results[r]<auc[0])
                ensemble_compare &= ind_results[r][0]<auc[0]
            print('ENSEMBLE w.', tx, ' better? ', ensemble_compare, '\n')
            ensemble_dct[tx] = 'Y' if ensemble_compare else 'N'
        all_lst.append(ensemble_dct)
    pd.DataFrame(all_lst).to_csv(output_file)



def athaliana_integ_predict(network_file, output_file,
                           train_dir, colFeats, output_image,
                           pos_ratio=0.5, neg_ratio=0.5, scale_pos_weight=None,
                           random_seed=5):
    a, df_in, df_out =  athaliana_integ_train(train_dir, colFeats, output_image,
                                              pos_ratio, neg_ratio, scale_pos_weight,
                                              random_seed)
    clf, index, scaler, tauroc, taupr = a
    if network_file is not None:
        df_tis  = pd.read_csv(network_file, sep=',')
        df_tis = df_tis.fillna(0)
        print("Loaded Network file : ", network_file)
        pred_array = classifier_prediction(df_tis[colFeats], clf, index, scaler, 0)
        print("Finished prediction for Network file : ", network_file)
        df_tis = df_tis[['edge']+colFeats]
        df_tis['wt'] = pred_array
        df_tis.sort_values(by=['wt'], inplace=True, ascending=False)
        return df_tis, df_in, df_out, tauroc, taupr
    else:
        return None, df_in, df_out, tauroc, taupr

def athaliana_rankavg_predict(network_file, output_file, train_dir, colFeats):
    athaliana_rankavg_train(train_dir, colFeats)

def athaliana_unionavg_predict(network_file, output_file, train_dir, unionFeats):
    athaliana_unionavg_train(train_dir, unionFeats)

def athaliana_integ_predict7(network_file, output_file):
    # network and output file1
    train_dir = 'data/athaliana_raw/'
    colFeats = ["chemical-clr", "chemical-aracne", "chemical-grnboost", "chemical-mrnet","chemical-tinge","chemical-wgcna",
         "development-clr","development-aracne","development-grnboost","development-mrnet","development-tinge","development-wgcna",
         "flower-clr","flower-aracne","flower-grnboost","flower-mrnet","flower-tinge","flower-wgcna",
         "hormone-aba-iaa-ga-br-clr","hormone-aba-iaa-ga-br-aracne","hormone-aba-iaa-ga-br-grnboost","hormone-aba-iaa-ga-br-mrnet","hormone-aba-iaa-ga-br-tinge","hormone-aba-iaa-ga-br-wgcna",
         "hormone-ja-sa-ethylene-clr","hormone-ja-sa-ethylene-aracne","hormone-ja-sa-ethylene-grnboost","hormone-ja-sa-ethylene-mrnet","hormone-ja-sa-ethylene-tinge","hormone-ja-sa-ethylene-wgcna",
         "leaf-clr","leaf-aracne","leaf-grnboost","leaf-mrnet","leaf-tinge","leaf-wgcna",
         "light-clr","light-aracne","light-grnboost","light-mrnet","light-tinge","light-wgcna",
         "nutrients-clr","nutrients-aracne","nutrients-grnboost","nutrients-mrnet","nutrients-tinge","nutrients-wgcna",
         "root-clr","root-aracne","root-grnboost","root-mrnet","root-tinge","root-wgcna",
         "rosette-clr","rosette-aracne","rosette-grnboost","rosette-mrnet","rosette-tinge","rosette-wgcna",
         "seed-clr","seed-aracne","seed-grnboost","seed-mrnet","seed-tinge","seed-wgcna",
         "seedling1wk-clr","seedling1wk-aracne","seedling1wk-grnboost","seedling1wk-mrnet","seedling1wk-tinge","seedling1wk-wgcna",
         "seedling2wk-clr","seedling2wk-aracne","seedling2wk-grnboost","seedling2wk-mrnet","seedling2wk-tinge","seedling2wk-wgcna",
         "shoot-clr","shoot-aracne","shoot-grnboost","shoot-mrnet","shoot-tinge","shoot-wgcna",
         "stress-light-clr","stress-light-aracne","stress-light-grnboost","stress-light-mrnet","stress-light-tinge","stress-light-wgcna",
         "stress-other-clr","stress-other-aracne","stress-other-grnboost","stress-other-mrnet","stress-other-tinge","stress-other-wgcna",
         "stress-pathogen-clr","stress-pathogen-aracne","stress-pathogen-grnboost","stress-pathogen-mrnet","stress-pathogen-tinge","stress-pathogen-wgcna",
         "stress-salt-drought-clr","stress-salt-drought-aracne","stress-salt-drought-grnboost","stress-salt-drought-mrnet","stress-salt-drought-tinge","stress-salt-drought-wgcna",
         "stress-temperature-clr","stress-temperature-aracne","stress-temperature-grnboost","stress-temperature-mrnet","stress-temperature-tinge","stress-temperature-wgcna",
         "wholeplant-clr","wholeplant-aracne","wholeplant-grnboost","wholeplant-mrnet","wholeplant-tinge","wholeplant-wgcna"]
    output_image = output_file.split(".")[0]  + ".png"
    df_tis, df_in, df_out, tauroc, taupr = athaliana_integ_predict(network_file,
                                output_file, train_dir, colFeats, output_image)
    if output_file is not None and output_file.endswith("tsv"):
        df_out_file = output_file.replace(".tsv", "_df_out.tsv")
        df_in_file = output_file.replace(".tsv", "_df_in.tsv")
        df_tis.to_csv(output_file, sep="\t", index=False)
        df_in.to_csv(df_in_file, sep="\t", index=False)
        df_out.to_csv(df_out_file, sep="\t", index=False)
    if output_file is not None and output_file.endswith("csv"):
        df_out_file = output_file.replace(".csv", "_df_out.csv")
        df_in_file = output_file.replace(".csv", "_df_in.csv")
        df_tis.to_csv(output_file, index=False)
        df_in.to_csv(df_in_file, index=False)
        df_out.to_csv(df_out_file, index=False)


def athaliana_integ_predict8(network_file, output_file):
    # network and output file1
    train_dir = 'data/athaliana_raw/'
    colFeats = ["chemical-clr", "chemical-aracne", "chemical-grnboost", "chemical-tinge",
         "development-clr","development-aracne","development-grnboost","development-tinge",
         "flower-clr","flower-aracne","flower-grnboost","flower-tinge",
         "hormone-aba-iaa-ga-br-clr","hormone-aba-iaa-ga-br-aracne","hormone-aba-iaa-ga-br-grnboost","hormone-aba-iaa-ga-br-tinge",
         "hormone-ja-sa-ethylene-clr","hormone-ja-sa-ethylene-aracne","hormone-ja-sa-ethylene-grnboost","hormone-ja-sa-ethylene-tinge",
         "leaf-clr","leaf-aracne","leaf-grnboost","leaf-tinge",
         "light-clr","light-aracne","light-grnboost","light-tinge",
         "nutrients-clr","nutrients-aracne","nutrients-grnboost","nutrients-tinge",
         "root-clr","root-aracne","root-grnboost","root-tinge",
         "rosette-clr","rosette-aracne","rosette-grnboost","rosette-tinge",
         "seed-clr","seed-aracne","seed-grnboost","seed-tinge",
         "seedling1wk-clr","seedling1wk-aracne","seedling1wk-grnboost","seedling1wk-tinge",
         "seedling2wk-clr","seedling2wk-aracne","seedling2wk-grnboost","seedling2wk-tinge",
         "shoot-clr","shoot-aracne","shoot-grnboost","shoot-tinge",
         "stress-light-clr","stress-light-aracne","stress-light-grnboost","stress-light-tinge",
         "stress-other-clr","stress-other-aracne","stress-other-grnboost","stress-other-tinge",
         "stress-pathogen-clr","stress-pathogen-aracne","stress-pathogen-grnboost","stress-pathogen-tinge",
         "stress-salt-drought-clr","stress-salt-drought-aracne","stress-salt-drought-grnboost","stress-salt-drought-tinge",
         "stress-temperature-clr","stress-temperature-aracne","stress-temperature-grnboost","stress-temperature-tinge",
         "wholeplant-clr","wholeplant-aracne","wholeplant-grnboost","wholeplant-tinge"]
    output_image = output_file.split(".")[0]  + ".png"
    df_tis, df_in, df_out, tauroc, taupr = athaliana_integ_predict(network_file,
                                output_file, train_dir, colFeats, output_image)
    if output_file is not None and output_file.endswith("tsv"):
        df_out_file = output_file.replace(".tsv", "_df_out.tsv")
        df_in_file = output_file.replace(".tsv", "_df_in.tsv")
        df_tis.to_csv(output_file, sep="\t", index=False)
        df_in.to_csv(df_in_file, sep="\t", index=False)
        df_out.to_csv(df_out_file, sep="\t", index=False)
    if output_file is not None and output_file.endswith("csv"):
        df_out_file = output_file.replace(".csv", "_df_out.csv")
        df_in_file = output_file.replace(".csv", "_df_in.csv")
        df_tis.to_csv(output_file, index=False)
        df_in.to_csv(df_in_file, index=False)
        df_out.to_csv(df_out_file, index=False)



def athaliana_integ_predict9(network_file, output_file):
    # network and output file1
    train_dir = 'data/athaliana_raw/'
    colFeats = ["chemical-clr", "chemical-grnboost", 
         "development-clr","development-grnboost",
         "flower-clr","flower-grnboost",
         "hormone-aba-iaa-ga-br-clr","hormone-aba-iaa-ga-br-grnboost",
         "hormone-ja-sa-ethylene-clr","hormone-ja-sa-ethylene-grnboost",
         "leaf-clr","leaf-grnboost",
         "light-clr","light-grnboost",
         "nutrients-clr","nutrients-grnboost",
         "root-clr","root-grnboost",
         "rosette-clr","rosette-grnboost",
         "seed-clr","seed-grnboost",
         "seedling1wk-clr","seedling1wk-grnboost",
         "seedling2wk-clr","seedling2wk-grnboost",
         "shoot-clr","shoot-grnboost",
         "stress-light-clr","stress-light-grnboost",
         "stress-other-clr","stress-other-grnboost",
         "stress-pathogen-clr","stress-pathogen-grnboost",
         "stress-salt-drought-clr","stress-salt-drought-grnboost",
         "stress-temperature-clr","stress-temperature-grnboost",
         "wholeplant-clr","wholeplant-grnboost"]
    output_image = output_file.split(".")[0]  + ".png"
    df_tis, df_in, df_out, tauroc, taupr = athaliana_integ_predict(network_file,
                                output_file, train_dir, colFeats, output_image)
    if output_file is not None and output_file.endswith("tsv"):
        df_out_file = output_file.replace(".tsv", "_df_out.tsv")
        df_in_file = output_file.replace(".tsv", "_df_in.tsv")
        df_tis.to_csv(output_file, sep="\t", index=False)
        df_in.to_csv(df_in_file, sep="\t", index=False)
        df_out.to_csv(df_out_file, sep="\t", index=False)
    if output_file is not None and output_file.endswith("csv"):
        df_out_file = output_file.replace(".csv", "_df_out.csv")
        df_in_file = output_file.replace(".csv", "_df_in.csv")
        df_tis.to_csv(output_file, index=False)
        df_in.to_csv(df_in_file, index=False)
        df_out.to_csv(df_out_file, index=False)


def athaliana_integ_predict10(network_file, output_file, options_file):
    # network and output file1
    train_dir = 'data/athaliana_raw/'
    with open(options_file) as jfptr:
        jsx = json.load(jfptr)
    output_image = jsx["OUTPUT_FILE"].split(".")[0]  + ".png"
    pos_range = jsx["POS_RANGE"]
    neg_range = jsx["NEG_RANGE"]
    ropt = [(p, q, None) for p in pos_range for q in neg_range]
    for p in jsx["POS_RANGE"]:
        for q in jsx["NEG_RANGE"]:
            for r in  jsx["POS_WEIGHT_RANGE"]:
                ropt.append((float(p), float(q), float(r))) 
    colFeats = jsx["COL_FEATS"]
    if ("RANDOM_SEED" in jsx) and jsx["RANDOM_SEED"] == "None":
        print("Random seed is None")
        random_seed = None
    else:
        random_seed = 5
    #print(ropt)
    oelts = ["TP-FP", "PCT POS. INCL.", "PCT NEG INCL.",
             "SCALE POS WT", "NUM EDGES",
             "TOTAL EDGES", "TRAIN SET",  "AUROC", "AUPR", "TEST SET",
             "TRAIN POSITIVES", "TRAIN NEGATIVES",
             "TRAIN TP", "TEST TP", "TOTAL TP",
             "TEST FP", "TRAIN FP", "TOTAL FP"]
    print("\t".join([str(x) for x in oelts]))
    for p,q,r in ropt:
        pos_ratio = float(p)
        neg_ratio = float(q)
        scale_pos_weight = r if r is None else float(r)
        df_tis, df_in, df_out, tauroc, taupr = athaliana_integ_predict(network_file, output_file,
                                        train_dir, colFeats, output_image,
                                        pos_ratio, neg_ratio, scale_pos_weight)
        for nx in jsx["OUT_EDGES"]: 
            df_wt = df_tis[['edge', 'wt']].head(n=int(nx))
            df_in_edge = pd.merge(df_in[['edge', 'prediction']], df_wt, how='inner', on=['edge'])
            df_out_edge = pd.merge(df_out[['edge', 'prediction']], df_wt, how='inner', on=['edge'])
            in_tp = df_in_edge.loc[df_in_edge.prediction == 1].shape[0]
            out_tp = df_out_edge.loc[df_out_edge.prediction == 1].shape[0]
            in_fp = df_in_edge.loc[df_in_edge.prediction == 0].shape[0]
            out_fp = df_out_edge.loc[df_out_edge.prediction == 0].shape[0]
            tp = in_tp + out_tp
            fp = in_fp + out_fp
            oelts = ["TP-FP", pos_ratio*100, neg_ratio*100, 
                     None if scale_pos_weight is None else scale_pos_weight,
                     nx, df_tis.shape[0], df_in.shape[0], tauroc, taupr,
                     df_out.shape[0],
                     df_in.loc[df_in.prediction == 1].shape[0],
                     df_in.loc[df_in.prediction == 0].shape[0],
                     df_out.loc[df_out.prediction == 1].shape[0],
                     df_out.loc[df_out.prediction == 0].shape[0],
                     in_tp, out_tp, tp,  in_fp, out_fp, fp]
            print("\t".join([str(x) for x in oelts]))


def athaliana_integ_predict11(network_file, output_file, options_file):
    # network and output file1
    train_dir = 'data/athaliana_raw/'
    with open(options_file) as jfptr:
        jsx = json.load(jfptr)
    output_image = jsx["OUTPUT_FILE"].split(".")[0]  + ".png"
    colFeats = jsx["COL_FEATS"]
    athaliana_rankavg_predict(network_file, output_file, train_dir, colFeats)

def athaliana_integ_predict12(network_file, output_file, options_file):
    train_dir = 'data/athaliana_raw/'
    with open(options_file) as jfptr:
        jsx = json.load(jfptr)
    output_image = jsx["OUTPUT_FILE"].split(".")[0]  + ".png"
    unionFeats = jsx["UNION_FEATS"]
    athaliana_unionavg_predict(network_file, output_file, train_dir, unionFeats)


def athaliana_integ_predict15(network_file, output_file, options_file):
    # network and output file1
    train_dir = 'data/athaliana_raw/'
    with open(options_file) as jfptr:
        jsx = json.load(jfptr)
    output_image = jsx["OUTPUT_FILE"].split(".")[0]  + ".png"
    pos_range = jsx["POS_RANGE"]
    neg_range = jsx["NEG_RANGE"]
    ropt = [(p, q, None) for p in pos_range for q in neg_range]
    for p in jsx["POS_RANGE"]:
        for q in jsx["NEG_RANGE"]:
            for r in  jsx["POS_WEIGHT_RANGE"]:
                ropt.append((float(p), float(q), float(r))) 
    colFeats = jsx["COL_FEATS"]
    if ("RANDOM_SEED" in jsx) and jsx["RANDOM_SEED"] == "None":
        print("Random seed is None")
        random_seed = None
    else:
        random_seed = 5
    if ("NROUNDS" in jsx):
        nrounds = int(jsx["NROUNDS"])
    #print(ropt)
    oelts = ["TP-FP", "PCT POS. INCL.", "PCT NEG INCL.",
             "SCALE POS WT",  "TRAIN SET",  "AUROC", "AUPR", "TEST SET",
             "TRAIN POSITIVES", "TRAIN NEGATIVES"]
    print("\t".join([str(x) for x in oelts]))
    for p,q,r in ropt:
        pos_ratio = float(p)
        neg_ratio = float(q)
        scale_pos_weight = r if r is None else float(r)
        aupr_lst = [0.0 for _ in range(nrounds)]
        auroc_lst = [0.0 for _ in range(nrounds)]
        for i in range(nrounds):
            _, df_in, df_out, tauroc, taupr = athaliana_integ_predict(None, output_file,
                                          train_dir, colFeats, output_image,
                                          pos_ratio, neg_ratio, scale_pos_weight,
                                          random_seed)
            aupr_lst[i] = taupr
            auroc_lst[i] = tauroc
            oelts = ["TP-FP", pos_ratio*100, neg_ratio*100, 
                  None if scale_pos_weight is None else scale_pos_weight,
                  df_in.shape[0], tauroc, taupr,
                  df_out.shape[0],
                  df_in.loc[df_in.prediction == 1].shape[0],
                  df_in.loc[df_in.prediction == 0].shape[0],
                  df_out.loc[df_out.prediction == 1].shape[0],
                  df_out.loc[df_out.prediction == 0].shape[0]]
            print("\t".join([str(x) for x in oelts]))
        oelts = ["TP-FP-AVG", pos_ratio*100, neg_ratio*100, 
                  None if scale_pos_weight is None else scale_pos_weight,
                  "-", 
                  str(np.mean(auroc_lst)) + "+/-" + str(np.std(auroc_lst)), 
                  str(np.mean(aupr_lst)) + "+/-" + str(np.std(aupr_lst)),
                  "-",
                  "-", 
                  "-",
                  "-",
                  "-"]
        print("\t".join([str(x) for x in oelts]))


 
if __name__ == "__main__":
    PROG_DESC = """Train with a subset of network and predit output """
    PARSER = argparse.ArgumentParser(description=PROG_DESC)
    PARSER.add_argument("run_type", help=""" Run Type""")
    PARSER.add_argument("-n", "--network_file",
                        default="edge_networks/union-all-networks.csv",
                        help="""network build from a reverse engineering methods
                                (currenlty supported: eda, adj, tsv)""")
    PARSER.add_argument("-o", "--output_file", default=None,
                        help="""Output File""")
    PARSER.add_argument("-p", "--options_file", default=None,
                        help="""Options File""")
    ARGS = PARSER.parse_args()
    run_type = ARGS.run_type
    print("Network file : ", ARGS.network_file)
    print("Run type : ",  run_type)
    print("Output file : ", ARGS.output_file)
    print("Options file : ", ARGS.options_file)
    if run_type == '1':
        athaliana_ensemble_predict1(ARGS.network_file, ARGS.output_file)
    elif run_type == '2':
        athaliana_ensemble_predict2(ARGS.network_file, ARGS.output_file)
    elif run_type == '3':
        athaliana_ensemble_predict3(ARGS.network_file, ARGS.output_file)
    elif run_type == '4':
        athaliana_ensemble_predict4(ARGS.network_file, ARGS.output_file)
    elif run_type == '5a':
        athaliana_ensemble_predict5a(ARGS.network_file, ARGS.output_file)
    elif run_type == '5b':
        athaliana_ensemble_predict5b(ARGS.network_file, ARGS.output_file)
    elif run_type == '6':
        athaliana_ensemble_predict6(ARGS.network_file, ARGS.output_file)
    elif run_type == '7':
        # Integration prediction with all methods
        athaliana_integ_predict7(ARGS.network_file, ARGS.output_file)
    elif run_type == '8':
        # Integration prediction with top 4 methods
        athaliana_integ_predict8(ARGS.network_file, ARGS.output_file)
    elif run_type == '9':
        # Integration prediction with top 2 methods
        athaliana_integ_predict9(ARGS.network_file, ARGS.output_file)
    elif run_type == '10':
        # Grid Integration prediction with all methods
        athaliana_integ_predict10(ARGS.network_file, ARGS.output_file, ARGS.options_file)
    elif run_type == '11':
        # Rank avg. prediction with all methods
        athaliana_integ_predict11(ARGS.network_file, ARGS.output_file, ARGS.options_file)
    elif run_type == '12':
        # Rank avg. after union of all methods
        athaliana_integ_predict12(ARGS.network_file, ARGS.output_file, ARGS.options_file)
    elif run_type == '13':
        analysis_data_v3_xgboost()
    elif run_type == '14':
        analysis_data_v4_xgboost()
    elif run_type == '14b':
        analysis_data_v5_xgboost()
    elif run_type == '15':
        # Grid Integration prediction with mutliple rounds
        athaliana_integ_predict15(ARGS.network_file, ARGS.output_file, ARGS.options_file)
    elif run_type == '16':
        athaliana_ensemble_predict7(ARGS.output_file)
    else:
        print("Invalid arguments")

