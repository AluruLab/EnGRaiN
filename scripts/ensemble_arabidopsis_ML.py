#!/usr/bin/env python
# coding: utf-8

# In[1]:

import argparse
import pandas as pd
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


def trainXGB(Xtrain, ytrain, Xtest, ytest, colFeats):
    dtrain = xgb.DMatrix(Xtrain,label=ytrain)
    dtest = xgb.DMatrix(Xtest,label=ytest)
    print('Setting XGB params')
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

#    param['booster'] = 'gblinear' #'dart' #'gblinear' # default is tree booster
#    param['lambda'] = 1
#    param['alpha'] = 1

    num_round = 220#60
    print('training the XGB classifier')
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
    plt.gcf().savefig('feature_importance_xgb.png')
    plt.show()
#    print 'Saving the models'
#    bst.save_model(name+'_xgb_v'+clf_VERSION+'.model')
#    bst.dump_model(name+'_xgb_v'+clf_VERSION+'_dump.raw.txt')
#    bst.dump_model(name+'_xgb_v'+clf_VERSION+'_dump.raw.txt',name+'_xgb_v'+clf_VERSION+'_featmap.txt')
    return bst


def classifier_train(X, y, runXGB, Xtest, ytest, colFeats, pca_comp = 0): # pca = 0 means no PCA applied
    print('Normalising the input data...')
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
        print('Running the XGB classifier')
        clf = trainXGB(pca_scaledX, y, scaler.transform(Xtest), ytest, colFeats)
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


def runRankAvg(data, methods):
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
    # reverse sorting the avgRank as auc needs scores.
    return get_auc(data['prediction'], df_edges['rankAvg'][::-1])

def individual_method(data, rankAvg=False):
    methods = [c for c in data.columns if c not in ['prediction', 'edge']]
    results = {}
    for m in methods:
        auc = get_auc(data['prediction'], data[m])
        results[m] = auc

    # Adding the rank Avg method
    if rankAvg:
        results['rankAvg'] = runRankAvg(data, methods)
    #print(results)
    for r in results:
        print(r, results[r])
    return results

def pandas_classifier(df_train, df_test, runXGB, K=5):
    print('Performing ' + str(K) + '-fold cross validation')
    auc_fold = []
    colFeats = [c for c in df_train.columns if c not in ['prediction', 'edge']]
    print(colFeats)

    for k in range(K):# performing K fold validation
        #if k == 0: # running only for k'th fold
            print('Fold_num = ' + str(k))
            #train_rows = [i for i in range(len(df_train)) if i%K!=k]
            datatrain = df_train.loc[[i for i in range(len(df_train)) if i%K!=k]] # training
            #valid_rows = [i for i in range(len(df_train)) if i%K==k]
            datavalid = df_train.loc[[i for i in range(len(df_train)) if i%K==k]] # taking every k'th example
#             Xtrain =  #.iloc[:, 0:-1]
#             ytrain =  #.iloc[:, -1]
#             Xvalid =  #.iloc[:, 0:-1]
#             yvalid = #.iloc[:, -1]
            print('--------------------------------------------------------------')
            print('Calling the classifier to train')
            scaler, pca, clf, index = classifier_train(datatrain[colFeats], datatrain['prediction'],
                    runXGB, datavalid[colFeats], datavalid['prediction'],
                    colFeats)
            print('Analysing the test predictions for fold num ', k)
            pred_array, auc = classifier_test(datavalid[colFeats],
                    datavalid['prediction'], clf, index, scaler, 0)
            auc_fold.append(auc[0])
            print('test auc = '+str(auc[0]) )
            individual_method(datavalid, rankAvg=False)
            print('------------------------------------------------------------')
    if K != 0:
        print('************************************************************************')
        print(auc_fold)#, sum(np.array(auc_fold))/int(K))
        print('Average '+str(K)+' fold CV result= ', str(sum(np.array(auc_fold))/int(K)))
        print('************************************************************************')

#     pred_array, auc = classifier_test(df_test[colFeats], df_test['prediction'], clf, index, scaler, 0)
#     print('TEST AUC on standalone data = ', auc[0])
#     print('individual methods: ', individual_method(df_test))
    return [clf, index, scaler]

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


def analysis_data_v3_xgboost():
    df_train, df_test = load_v3_dataset()
    # ## Binary Classification using traditional ML techniques - XGBoost

    # In[2]:


    # In[15]:


    # Convert data to pandas before passing
    # np.random.seed(15)
    # random.seed(15)
    # df_X = pd.DataFrame(X, columns=[c for c in df.columns if c!='prediction'])
    # df_y = pd.DataFrame(pd.Series(y, name='label'))
    # df_train = pd.concat([df_X, df_y], axis=1)
    # df_test

    #print(df_classifier, df)
    #print(df.columns)

    trained_params = pandas_classifier(df_train, df_test, 1)


    # In[17]:


    print('individual methods on test: ', individual_method(df_test, rankAvg=True))


    # ### Loading the arabidopsis data

    # In[13]:


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
    clf, index, scaler = trained_params

    print('Check col order: ', colFeats)
    pred_array, auc = classifier_test(df_abd[colFeats], df_abd['prediction'], clf, index, scaler, 0)
    print('ENSEMBLE AUC on Arabidopsis data = ', auc[0])
    print('individual methods on arabidopsis: ')
    individual_method(df_abd)


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
    trained_params = pandas_classifier(df_train, df_test, 1)
    ath_types = ['chemical', 'development', 'flower', 'hormone-aba-iaa-ga-br' ,'hormone-ja-sa-ethylene',
            'leaf', 'light', 'nutrients', 'root', 'rosette', 'seed', 'seedling1wk', 'seedling2wk',
            'shoot', 'stress-light', 'stress-other', 'stress-pathogen', 'stress-salt-drought',
            'stress-temperature', 'wholeplant']

    # predicting using the trained params
    clf, index, scaler = trained_params

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


def athaliana_ensemble_train(train_dir, tissue_types_train, colFeats):
    # training
    df_tis_train = pd.DataFrame([])
    for i, t in enumerate(tissue_types_train):
        print(train_dir, i, t)
        tisOnes  = pd.read_csv(train_dir+'/'+t+'-positives.csv', sep=',')
        tisOnes['prediction'] = 1
        tisZeros = pd.read_csv(train_dir+'/'+t+'-negatives.csv', sep=',')
        tisZeros['prediction'] = 0
        df_tis= pd.concat([tisOnes, tisZeros], ignore_index=True)
        # # Imputation step with zeros
        df_tis = df_tis.fillna(0)
        df_tis_train = pd.concat([df_tis_train, df_tis], ignore_index=True)

    trained_params_tissue = pandas_classifier(df_tis_train, df_tis_train, 1)
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


    clf, index, scaler = athaliana_ensemble_train(train_dir, tissue_types_train, colFeats)

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
        ind_results = individual_method(df_tis, rankAvg=False)
        ensemble_compare = True
        for r in ind_results:
            #print(r, ind_results[r]<auc[0])
            ensemble_compare &= ind_results[r][0]<auc[0]
        print('\n ENSEMBLE better? ', ensemble_compare, '\n')


def athaliana_ensemble_predict(network_file, output_file, tissue_types=None):
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


    clf, index, scaler = athaliana_ensemble_train(train_dir, tissue_types_train, colFeats)

    df_tis  = pd.read_csv(network_file, sep=',')
    df_tis = df_tis.fillna(0)
    print("Loaded Network file : ", network_file)
    pred_array = classifier_prediction(df_tis[colFeats], clf, index, scaler, 0)
    print("Finished prediction for Network file : ", network_file)
    df_tis['wt'] = pred_array
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


    clf, index, scaler = athaliana_ensemble_train(train_dir, tissue_types_train, colFeats)

    df_tis  = pd.read_csv(network_file, sep=',')
    df_tis = df_tis.fillna(0)
    print("Loaded Network file : ", network_file)
    pred_array = classifier_prediction(df_tis[colFeats], clf, index, scaler, 0)
    print("Finished prediction for Network file : ", network_file)
    df_tis['wt'] = pred_array
    df_tis.to_csv(output_file, sep="\t", index=False)


if __name__ == "__main__":
    PROG_DESC = """Train with a subset of network and predit output """
    PARSER = argparse.ArgumentParser(description=PROG_DESC)
    PARSER.add_argument("network_file", 
                        help="""network build from a reverse engineering methods
                                (currenlty supported: eda, adj, tsv)""")
    PARSER.add_argument("output_file",
                        help="""Output File""")
    ARGS = PARSER.parse_args()
    athaliana_ensemble_predict2(ARGS.network_file, ARGS.output_file)

