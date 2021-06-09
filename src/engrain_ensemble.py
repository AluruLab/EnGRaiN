import argparse
import pandas as pd
import json
import functools as ft
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
import random
import operator, itertools
##
from pandas import ExcelWriter
from pandas import ExcelFile
##
from sklearn import svm
from sklearn import manifold
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import TSNE
from sklearn import model_selection
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
#
from collections import Counter
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import NullFormatter

import warnings
warnings.filterwarnings('ignore')
#

def get_auc_plot(y, scores):
    y = np.array(y).astype(int)
    fpr, tpr, thresholds = metrics.roc_curve(y, scores)
    # print(fpr, tpr, thresholds)
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
#?
#       print('For threshold : ', th, '->', ' Accuracy ', accuracy_array[i],
#             ' Sensitivity ', sensitivity_array[i], 'specificity ',
#             specificity_array[i])
#       show_confusion_matrix(y, pred_array)
#     print('accuracy_array', accuracy_array)
#     print('sensitivity_array', sensitivity_array)
#     print('specificity_array', specificity_array)
#     print('thresholds_array', thresholds)
#
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



def trainXGB(Xtrain, ytrain, Xtest, ytest, colFeats,
             output_image=None, scale_pos_weight=None):
    dtrain = xgb.DMatrix(Xtrain,label=ytrain)
    dtest = xgb.DMatrix(Xtest,label=ytest)
    #print('Setting XGB params')
    evallist  = [(dtest,'test'), (dtrain,'train')]

    param = {}
    # use binary logistic
    param['objective'] = 'binary:logistic' #'multi:softprob'
    # scale weight of positive examples
    param['eta'] = 0.01
    param['max_depth'] = 7
    param['gamma'] = 0
    #  param['silent'] = 1
    param['nthread'] = 6
    #param['subsample'] = 0.5#0.7 # number of examples for 1 tree (subsampled from total)
    #param['colsample_bytree'] = 0.5#0.7 # ratio of columns for training 1 tree
    #param['num_class'] = NUM_CLASS
    param['eval_metric'] = 'auc'#'mlogloss' #auc

    # CLASS Imbalance handling!
    # param['scale_pos_weight'] = 10# sum(negative cases) / sum(positive cases)
    if scale_pos_weight is not None:
        param['scale_pos_weight'] = scale_pos_weight

    # param['booster'] = 'gblinear' #'dart' #'gblinear' # default is tree booster
    # param['lambda'] = 1
    # param['alpha'] = 1

    num_round = 220#60
    # print('training the XGB classifier')
    bst = xgb.train(param, dtrain, num_round, evallist, 
                    early_stopping_rounds=100, verbose_eval=False)
    # print('training completed, printing the relative importance: \
    #        (feature id: importance value)')
    importance = bst.get_fscore()
    importance = sorted(importance.items(), key=operator.itemgetter(1))

    # print(importance)

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
    # plt.show()
    # print 'Saving the models'
    # bst.save_model(name+'_xgb_v'+clf_VERSION+'.model')
    # bst.dump_model(name+'_xgb_v'+clf_VERSION+'_dump.raw.txt')
    # bst.dump_model(name+'_xgb_v'+clf_VERSION+'_dump.raw.txt',name+'_xgb_v'+clf_VERSION+'_featmap.txt')
    return bst


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
    return [auroc, aupr]

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
    # print(adf, adf['prediction'].isna().sum(),
    #       np.isinf(adf['scaleLSum']).sum())
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


def individual_method(data, methods, rank_avg=False, print_rocpr=True):
    #
    # Methods
    #
    results = {}
    for m in methods:
        auc = get_auc(data['prediction'], data[m])
        results[m] = auc
    #
    # Adding the rank Avg method
    # 
    if rank_avg is True:
        results['RankAvg-Sorted'] = runRankAvg1(data, methods)
        results['RankAvg-Reciprocal'] = runRankAvg2(data, methods)
        #results['RankAvg-UpDown'] = runRankAvg3(data, methods)
        results['RankAvg-ScaleSum'] = scalesum_stats(data, methods)
    # Results
    if print_rocpr is True:
        for r in results:
            print(r, results[r])
    return results


def classifier_train(X, y, Xtest, ytest, colFeats,
        output_image=None, scale_pos_weight=None, 
        pca_comp = 0): # pca = 0 means no PCA applied
    # print('Normalising the input data...')
    # print(X, y)
    scaler = StandardScaler() #MinMaxScaler() #StandardScaler()
    scaler.fit(X)
    scaledX = scaler.transform(X)
    if pca_comp is True:
        pca = PCA(n_components = pca_comp)
        pca.fit(scaledX)
        pca_scaledX = pca.transform(scaledX)
    else:
        pca_scaledX = scaledX
        pca = None
    # print('Running the XGB classifier')
    clf = trainXGB(pca_scaledX, y, scaler.transform(Xtest), ytest,
                   colFeats, output_image, scale_pos_weight)
    return scaler, pca, clf



def classifier_test(X, y, clf, scaler, pca):
    scaledX = scaler.transform(X)
    pca_scaledX = scaledX
    if pca is not None:
        pca_scaledX = pca.transform(scaledX)
    pca_scaledXG = xgb.DMatrix(pca_scaledX, label=y)
    pred_array = clf.predict(pca_scaledXG, ntree_limit=clf.best_iteration)
    #.reshape(y.shape[0], NUM_CLASS)#, ntree_limit=clf.best_iteration)
    scores = pred_array
    auc = get_auc_plot(y, scores)
    # Compute confusion matrix
    #show_confusion_matrix(y, pred_array)
    return pred_array, auc # error



def pandas_classifier(df_train, df_test, colFeats,
                      output_image=None, scale_pos_weight=None, 
                      K=10, rank_avg=False, ind_fold_auc=True,
                      print_rocpr=True):
    print('Performing ' + str(K) + '-fold cross validation with ', colFeats)
    auc_fold = []
    pr_fold = []
    # performing K fold validation
    for k in range(K): 
        # training
        datatrain = df_train.loc[[i for i in range(len(df_train)) if i%K!=k]]
        # taking every k'th example for validation
        datavalid = df_train.loc[[i for i in range(len(df_train)) if i%K==k]] 
        scaler, pca, clf = classifier_train(datatrain[colFeats],
            datatrain['prediction'], datavalid[colFeats], 
            datavalid['prediction'], colFeats, output_image, scale_pos_weight)
        pred_array, auc = classifier_test(datavalid[colFeats],
            datavalid['prediction'], clf, scaler, None)
        auc_fold.append(auc[0])
        pr_fold.append(auc[1])
        if ind_fold_auc is True:
            individual_method(datavalid, colFeats, rank_avg=rank_avg)
    avg_auc = sum(np.array(auc_fold))/int(K) if K > 0 else 0.0
    avg_aupr = sum(np.array(pr_fold))/int(K) if K > 0 else 0.0
    std_auc = np.std(np.array(auc_fold)) if K > 0 else 0.0
    std_aupr = np.std(np.array(pr_fold)) if K > 0 else 0.0
    if print_rocpr is True:
        print('************************************************************************')
        #print(auc_fold)#, sum(np.array(auc_fold))/int(K))
        #print(pr_fold)
        print('AVG ', str(K), 'AUROC & AUPR', str(avg_auc) + "(" + str(std_auc) + ")",
              str(avg_aupr), "(", str(std_aupr), ")")
        print('************************************************************************')

    # pred_array, auc = classifier_test(df_test[colFeats], 
    #                       df_test['prediction'], clf, scaler)
    # print('TEST AUC on standalone data = ', auc[0])
    # print('individual methods: ', individual_method(df_test))
    return [clf, scaler, avg_auc, std_auc, avg_aupr, std_aupr]



def sim_load_dataset(edge_weights_file):
    # In[6]:
    df = pd.read_csv(edge_weights_file, sep=',')
    del df['pcc']
    del df['irafnet']
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


def sim_learn_classify_test(df_train, df_test, allOnes, allZeros,
                            colFeats, output_image):
    sim_results = {}
    nfeat = str(len(colFeats))
    _,_, auroc, sroc, aupr, spr = pandas_classifier(df_train, df_test, 
            colFeats, output_image, ind_fold_auc=False)
    sim_results['ENGRAIN'] = [auroc, aupr]
    sim_results['STD_ENGRAIN'] = [sroc, spr]
    ind_rpr = individual_method(df_test, colFeats, rank_avg=True)
    sim_results.update(ind_rpr)
    sim_results['ScaleSum-'+nfeat] = runScaleSum(allOnes, allZeros, colFeats)
    sim_results['ScaleLSum-'+nfeat] = runScaleLSum(allOnes, allZeros, colFeats)
    print(sim_results)
    return sim_results

def sim_analyse(df_train, df_test, allOnes, allZeros,
                output_image_pfx):
    colFeats = [c for c in df_train.columns if c not in ['prediction', 'edge']]
    output_image = output_image_pfx + "-full.png"
    sim_results = sim_learn_classify_test(df_train, df_test, allOnes, allZeros,
                                          colFeats, output_image)
    colFeats7 = ["clr", "grnboost", "aracne", "mrnet", "tinge", "wgcna", "genie3"]
    output_image = output_image_pfx + "-top7.png"
    sim_results7 = sim_learn_classify_test(df_train, df_test, allOnes, allZeros,
                                           colFeats7, output_image)
    sim_results7 = {"Top7-"+x:y for x,y in sim_results7.items()}
    sim_results.update(sim_results7)
    return pd.DataFrame.from_dict(sim_results, orient='index',
                columns=['AUROC', 'AUPR'])

def sim_load_v5_dataset():
    return sim_load_dataset('data/yeast-edge-weights-v5.csv')

def sim_load_v4_dataset():
    return sim_load_dataset('data/yeast-edge-weights-v4.csv')

def sim_load_v3_dataset():
    return sim_load_dataset('data/yeast-edge-weights-v3.csv')

def sim_analysis_data_v5():
    df_train, df_test, allOnes, allZeros = sim_load_v5_dataset()
    print("Loaded v5 dataset ")
    dfx = sim_analyse(df_train, df_test, allOnes, allZeros, "v5-analysis")
    print(dfx)

def sim_2k_analysis():
    files_2k  ={
      "250"  : "../data/yeast-edges-weights-2000.250.csv",
      "500"  : "../data/yeast-edges-weights-2000.500.csv",
      "750"  : "../data/yeast-edges-weights-2000.750.csv",
      "1000" : "../data/yeast-edges-weights-2000.1000.csv",
      "1250" : "../data/yeast-edges-weights-2000.1250.csv",
      "1500" : "../data/yeast-edges-weights-2000.1500.csv",
      "1750" : "../data/yeast-edges-weights-2000.1750.csv",
      "2000" : "../data/yeast-edges-weights-2000.2000.csv"
      }
    df_2k = pd.DataFrame()
    out_file = "sim_2k_engrain.csv"
    for kx, fy in files_2k.items():
        df_train, df_test, allOnes, allZeros = sim_load_dataset(fy)
        print("Loaded dataset : ", kx, fy)
        dfx = sim_analyse(df_train, df_test, allOnes, allZeros,
                          "m" + kx + "-2kanalysis")
        df_2k["AUROC-" + kx] = dfx["AUROC"]
        df_2k["AUPR-" + kx] = dfx["AUPR"]
    df_2k.to_csv(out_file)

def sim_analysis(options_file, output_file):
    with open(options_file) as f:
        jsx = json.load(f)
    out_img_pfx = jsx["OUT_PREFIX"]
    sim_df = pd.DataFrame()
    for kx, fy in jsx["INPUT_FILES"].items():
        df_train, df_test, allOnes, allZeros = sim_load_dataset(fy)
        print("Loaded dataset : ", kx, fy)
        dfx = sim_analyse(df_train, df_test, allOnes, allZeros,
                          out_img_pfx + kx)
        sim_df["AUROC-" + kx] = dfx["AUROC"]
        sim_df["AUPR-" + kx] = dfx["AUPR"]
    sim_df.to_csv(out_file)


if __name__ == "__main__":
    PROG_DESC = """Train with a subset of network and predict Ensemble """
    PARSER = argparse.ArgumentParser(description=PROG_DESC)
    SUBPARSERS = PARSER.add_subparsers(dest="sub_cmd")

    # create the parser for the "simulated" command
    # Runs results for the simulated commands
    PARSER_SIM = SUBPARSERS.add_parser("sim", help='Run Simulated Datasets')
    PARSER_SIM.add_argument("options_file", help="""File w. Input Options""")
    PARSER_SIM.add_argument("-o", "--output_file", default=None,
                            help="""Output File""")

    # create the parser for the "athaliana" command
    PARSER_ATH = SUBPARSERS.add_parser("ath", help='Run A. thaliana Datasets')
    PARSER_ATH.add_argument("options_file", help="""File w. Input Options""")
    PARSER_ATH.add_argument("-o", "--output_file", default=None,
                            help="""Output File""")


    # create the parser for the "pred" command
    PARSER_PRED = SUBPARSERS.add_parser("pred", help='Output A. thaliana Predictions')
    PARSER_PRED.add_argument("options_file", help="""File w. Input Options""")
    PARSER_PRED.add_argument("-n", "--network_file",
                        default="edge_networks/union-all-networks.csv",
                        help="""network build from a reverse engineering methods
                                (network is a csv file)""")
    PARSER_PRED.add_argument("-o", "--output_file", default=None,
                         help="""Output File""")
    # Prase args
    ARGS = PARSER.parse_args()
    print(ARGS) 
    run_type = ARGS.sub_cmd
    print("Run Type : ",  run_type)
    print(" -> Option File : ",  ARGS.options_file)
    print(" -> Output File : ",  ARGS.output_file)
    if run_type == "sim":
        if ARGS.options_file == "v5":
            sim_analysis_data_v5()
        elif ARGS.options_file == "2k":
            sim_2k_analysis()
        else:
            sim_analysis(ARGS.options_file, ARGS.output_file)
    elif run_type == "ath":
        pass
    else:
        print(" -> Network File : ",  ARGS.network_file)
 
