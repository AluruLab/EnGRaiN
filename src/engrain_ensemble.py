import argparse
import pandas as pd
import json
import functools as ft
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
import random
import operator, itertools
#
#
from collections import Counter
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import NullFormatter
##
from pandas import ExcelWriter
from pandas import ExcelFile
##
from sklearn import svm
from sklearn import manifold
from sklearn import metrics
from sklearn import model_selection
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
#
import warnings
warnings.filterwarnings("ignore")
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
#       print("For threshold : ", th, "->", " Accuracy ", accuracy_array[i],
#             " Sensitivity ", sensitivity_array[i], "specificity ",
#             specificity_array[i])
#       show_confusion_matrix(y, pred_array)
#     print("accuracy_array", accuracy_array)
#     print("sensitivity_array", sensitivity_array)
#     print("specificity_array", specificity_array)
#     print("thresholds_array", thresholds)
#
    roc_auc = metrics.auc(fpr, tpr)
    plt.title("Receiver Operating Characteristic")
    plt.plot(fpr, tpr, "b", label = "AUC = %f" % roc_auc)
    plt.legend(loc = "lower right")
    plt.plot([0, 1], [0, 1],"r--")
    plt.xlim([-0.001, 1])
    plt.ylim([0, 1.001])
    plt.ylabel("True Positive Rate")
    plt.xlabel("False Positive Rate")
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
    #print("Setting XGB params")
    evallist  = [(dtest,"test"), (dtrain,"train")]

    param = {}
    # use binary logistic
    param["objective"] = "binary:logistic" #"multi:softprob"
    # scale weight of positive examples
    param["eta"] = 0.01
    param["max_depth"] = 7
    param["gamma"] = 0
    #  param["silent"] = 1
    param["nthread"] = 6
    #param["subsample"] = 0.5#0.7 # number of examples for 1 tree (subsampled from total)
    #param["colsample_bytree"] = 0.5#0.7 # ratio of columns for training 1 tree
    #param["num_class"] = NUM_CLASS
    param["eval_metric"] = "auc"#"mlogloss" #auc

    # CLASS Imbalance handling!
    # param["scale_pos_weight"] = 10# sum(negative cases) / sum(positive cases)
    if scale_pos_weight is not None:
        param["scale_pos_weight"] = scale_pos_weight

    # param["booster"] = "gblinear" #"dart" #"gblinear" # default is tree booster
    # param["lambda"] = 1
    # param["alpha"] = 1

    num_round = 220#60
    # print("training the XGB classifier")
    bst = xgb.train(param, dtrain, num_round, evallist, 
                    early_stopping_rounds=100, verbose_eval=False)
    # print("training completed, printing the relative importance: \
    #        (feature id: importance value)")
    importance = bst.get_fscore()
    importance = sorted(importance.items(), key=operator.itemgetter(1))

    # print(importance)

    # we will print from df1 dataframe, getting the corresponding feature names.
    df1 = pd.DataFrame(importance, columns=["feature", "fscore"])
    # Normalizing the feature scores
    df1["fscore"] = df1["fscore"] / df1["fscore"].sum()

    #print(df1)
    # adding a column of feature name
    # DEFINE as global
    #colFeats = [c for c in df_train.columns if c not in ["prediction", "edge"]]
    #colFeats = ["clr", "aracne", "grnboost", "mrnet", "tinge", "wgcna"]

    # column_names = df_train.columns[:-1]
    df1["feature_names"] = pd.Series([colFeats[int(f[0].replace("f", ""))] for f in importance])

    df1.plot()
    df1.plot(kind="barh", x="feature_names", y="fscore", legend=False, figsize=(6, 10))
    plt.title("XGBoost Feature Importance")
    plt.xlabel("relative importance")
    if output_image is None:
        plt.gcf().savefig("feature_importance_xgb.png")
    else:
        plt.gcf().savefig(output_image)
    # plt.show()
    # print "Saving the models"
    # bst.save_model(name+"_xgb_v"+clf_VERSION+".model")
    # bst.dump_model(name+"_xgb_v"+clf_VERSION+"_dump.raw.txt")
    # bst.dump_model(name+"_xgb_v"+clf_VERSION+"_dump.raw.txt",name+"_xgb_v"+clf_VERSION+"_featmap.txt")
    return bst


def runRankAvg1(data, methods):
    # get the ranks for the individual methods
    df_edges = pd.DataFrame(data["edge"])
    for i, m in enumerate(methods):
        rankM = pd.concat([data["edge"], data[m]], axis=1)
        # sort according to the method value
        rankM.sort_values(by=[m], inplace=True, ascending=False)
        rankM.reset_index(drop=True, inplace=True)
        rankNum = pd.Series([i for i in range(0, rankM.shape[0])], name="rank_m"+str(i))
        rankM = pd.concat([rankM, rankNum], axis=1)
        df_edges = pd.merge(left=df_edges, right=rankM, left_on="edge", right_on="edge")
        del df_edges[m]

    df_edges["rankAvg"] = df_edges.mean(axis=1)
    mx_rank = df_edges["rankAvg"]
    df_edges["rankAvg"] = 0 - df_edges["rankAvg"]
    # reverse sorting the avgRank as auc needs scores.
    #return get_auc(data["prediction"], df_edges["rankAvg"][::-1])
    return get_auc(data["prediction"], df_edges["rankAvg"])


def runRankAvg2(data, methods):
    # get the ranks for the individual methods
    df_edges = data[methods + ["prediction"]]
    df_edges["rankAvg"] = 1/df_edges[methods].rank(ascending=False).mean(axis=1)
    # reverse sorting the avgRank as auc needs scores.
    return get_auc(data["prediction"], df_edges["rankAvg"])


def runRankAvg3(data, methods):
    # get the ranks for the individual methods
    df_edges = data[methods + ["prediction"]]
    df_edges["rankAvg"] = df_edges[methods].rank(ascending=False).mean(axis=1)
    # reverse sorting the avgRank as auc needs scores.
    x, y = get_auc(1 - data["prediction"], df_edges["rankAvg"])
    return [x, y]

def scalesum_stats(adf, methods):
    acx = adf[methods]
    astd = acx.std(axis=0)
    amean = acx.mean(axis=0)
    sx1 = (adf[methods] - amean)/astd
    adf["wt"] =sx1.sum(axis=1)
    aupr = metrics.average_precision_score(adf.prediction, adf.wt)
    auroc = metrics.roc_auc_score(adf.prediction, adf.wt)
    return [auroc, aupr]

def scalelsum_stats(df, colFeats):
    adf = pd.concat([allOnes, allZeros])
    adf[["s", "t"]] = adf["edge"].str.split("-", expand=True)
    ax1 = adf[methods+["s"]].rename(columns={"s":"node"})
    ax2 = adf[methods+["t"]].rename(columns={"t":"node"})
    acx = pd.concat([ax1,ax2])
    astd = acx.groupby("node").std()
    amean = acx.groupby("node").mean()
    amean.reset_index()
    astd.reset_index()
    tx1 = pd.DataFrame(adf["s"]).rename(columns={"s":"node"}) 
    tx2 = pd.DataFrame(adf["t"]).rename(columns={"t":"node"})
    smean = tx1.merge(amean, on="node")  
    sstd = tx1.merge(astd, on="node")   
    tmean = tx2.merge(amean, on="node") 
    tstd = tx2.merge(astd, on="node")
    e1 = (adf[methods] - smean[methods])/sstd[methods]
    e2 = (adf[methods] - tmean[methods])/tstd[methods]
    ex1 = e1 **2
    ex2 = e2 **2
    sx1 = np.sqrt(ex1 + ex2)
    adf["wt"] =sx1.sum(axis=1)
    aupr = metrics.average_precision_score(adf.prediction, adf.wt)
    auroc = metrics.roc_auc_score(adf.prediction, adf.wt)
    return auroc, aupr


def runScaleLSum(allOnes, allZeros, methods):
    adf = pd.concat([allOnes, allZeros])
    adf.fillna(0)
    adf[["s", "t"]] = adf["edge"].str.split("-", expand=True)
    ax1 = adf[methods+["s"]].rename(columns={"s":"node"})
    ax2 = adf[methods+["t"]].rename(columns={"t":"node"})
    acx = pd.concat([ax1,ax2])
    astd = acx.groupby("node").std()
    amean = acx.groupby("node").mean()
    amean.reset_index()
    astd.reset_index()
    tx1 = pd.DataFrame(adf["s"]).rename(columns={"s":"node"}) 
    tx2 = pd.DataFrame(adf["t"]).rename(columns={"t":"node"})
    smean = tx1.merge(amean, on="node")  
    sstd = tx1.merge(astd, on="node")   
    tmean = tx2.merge(amean, on="node") 
    tstd = tx2.merge(astd, on="node")
    e1 = (adf[methods] - smean[methods])/sstd[methods]
    e2 = (adf[methods] - tmean[methods])/tstd[methods]
    ex1 = e1 **2
    ex2 = e2 **2
    sx1 = np.sqrt(ex1 + ex2)
    adf["scaleLSum"] =sx1.sum(axis=1)
    adf["scaleLSum"].replace(np.inf, 0, inplace=True)
    adf["scaleLSum"].replace(-np.inf, 0, inplace=True)
    # print(adf, adf["prediction"].isna().sum(),
    #       np.isinf(adf["scaleLSum"]).sum())
    return get_auc(adf["prediction"], adf["scaleLSum"])


def runScaleSum(allOnes, allZeros, methods):
    adf = pd.concat([allOnes, allZeros])
    acx = adf[methods]
    astd = acx.std(axis=0)
    amean = acx.mean(axis=0)
    sx1 = (adf[methods] - amean)/astd
    adf["wt"] =sx1.sum(axis=1)
    adf["scaleSum"] =sx1.sum(axis=1)
    return get_auc(adf["prediction"], adf["scaleSum"])


def individual_method(data, methods, rank_avg=False, print_rocpr=True):
    #
    # Methods
    #
    results = {}
    for m in methods:
        auc = get_auc(data["prediction"], data[m])
        results[m] = auc
    #
    # Adding the rank Avg method
    # 
    if rank_avg is True:
        results["RankAvg-Sorted"] = runRankAvg1(data, methods)
        results["RankAvg-Reciprocal"] = runRankAvg2(data, methods)
        #results["RankAvg-UpDown"] = runRankAvg3(data, methods)
        results["RankAvg-ScaleSum"] = scalesum_stats(data, methods)
    # Results
    if print_rocpr is True:
        for r in results:
            print(r, results[r])
    return results


def xgb_train(X, y, Xtest, ytest, colFeats,
        output_image=None, scale_pos_weight=None, 
        pca_comp = False): # pca = 0 means no PCA applied
    # print("Normalising the input data...")
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
    # print("Running the XGB classifier")
    clf = trainXGB(pca_scaledX, y, scaler.transform(Xtest), ytest,
                   colFeats, output_image, scale_pos_weight)
    return scaler, pca, clf

def ml_train(X, y, colFeats, method, output_image=None,  pca_comp = False): 
    # pca = False means no PCA applied
    #print("ML T: ", X.shape)
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
    clf = None
    if method == 'rf':
        clf =  RandomForestClassifier(max_depth=5, n_estimators=10, 
                                      max_features=1)
        clf.fit(pca_scaledX, y)
    elif method == 'svc':
        clf = SVC()
        clf.fit(pca_scaledX, y)
    elif method == 'mlp':
        clf = MLPClassifier(alpha=1, max_iter=1000,
                            solver='lbfgs', hidden_layer_sizes=(12,))
        clf.fit(pca_scaledX, y)
    elif method == 'sgd':
        clf = SGDClassifier(alpha=1e-5)
        clf.fit(pca_scaledX, y)
    return scaler, pca, clf
 
def classifier_train(X, y, Xtest, ytest, colFeats,
        output_image=None, scale_pos_weight=None, 
        pca_comp = False, method='xgb'): # pca = 0 means no PCA applied
    if method in ['svc', 'rf', 'mlp', 'sgd']:
        return ml_train(X, y, colFeats, method, output_image, pca_comp)
    else:
        return xgb_train(X, y, Xtest, ytest, colFeats, output_image,
                         scale_pos_weight, pca_comp)

def classifier_prediction(X, clf, scaler, pca, method):
    scaledX = scaler.transform(X)
    pca_scaledX = scaledX
    if pca is not None:
        pca_scaledX = pca.transform(scaledX)
    if method == 'xgb':
        pca_scaledXG = xgb.DMatrix(pca_scaledX)
        pred_array = clf.predict(pca_scaledXG, ntree_limit=clf.best_iteration)
    else:
        if hasattr(clf, "decision_function"):
            pred_array = clf.decision_function(pca_scaledX)
        else:
            pred_array = clf.predict_proba(pca_scaledX)
        if len(pred_array.shape) > 1 and pred_array.shape[1] == 2:
            pred_array = pred_array[:,1]
    return pred_array


def classifier_test(X, y, clf, scaler, pca, method):
    scaledX = scaler.transform(X)
    pca_scaledX = scaledX
    if pca is not None:
        pca_scaledX = pca.transform(scaledX)
    if method == 'xgb':
        pca_scaledXG = xgb.DMatrix(pca_scaledX, label=y)
        pred_array = clf.predict(pca_scaledXG, ntree_limit=clf.best_iteration)
    else:
        if hasattr(clf, "decision_function"):
            pred_array = clf.decision_function(pca_scaledX)
        else:
            pred_array = clf.predict_proba(pca_scaledX)
        #print("SH", pred_array.shape)
        if len(pred_array.shape) > 1 and pred_array.shape[1] == 2:
            pred_array = pred_array[:,1]
    #.reshape(y.shape[0], NUM_CLASS)#, ntree_limit=clf.best_iteration)
    scores = pred_array
    auc = get_auc_plot(y, scores)
    # Compute confusion matrix
    #show_confusion_matrix(y, pred_array)
    return pred_array, auc # error



def pandas_classifier(df_train, df_test, colFeats, output_image=None, 
                      method='xgb', scale_pos_weight=None, 
                      K=10, rank_avg=False, ind_fold_auc=True,
                      print_rocpr=True):
    print("Performing " + str(K) + "-fold cross validation with ", colFeats)
    auc_fold = []
    pr_fold = []
    # performing K fold validation
    for k in range(K): 
        # training
        datatrain = df_train.loc[[i for i in range(len(df_train)) if i%K!=k]]
        # taking every k"th example for validation
        datavalid = df_train.loc[[i for i in range(len(df_train)) if i%K==k]] 
        #print("DT :", datatrain.shape, datatrain[colFeats].shape)
        scaler, pca, clf = classifier_train(
            datatrain[colFeats], datatrain["prediction"], 
            datavalid[colFeats], datavalid["prediction"], 
            colFeats, output_image, scale_pos_weight, False, method)
        pred_array, auc = classifier_test(datavalid[colFeats],
            datavalid["prediction"], clf, scaler, None, method)
        auc_fold.append(auc[0])
        pr_fold.append(auc[1])
        if ind_fold_auc is True:
            individual_method(datavalid, colFeats, rank_avg=rank_avg)
    avg_auc = sum(np.array(auc_fold))/int(K) if K > 0 else 0.0
    avg_aupr = sum(np.array(pr_fold))/int(K) if K > 0 else 0.0
    std_auc = np.std(np.array(auc_fold)) if K > 0 else 0.0
    std_aupr = np.std(np.array(pr_fold)) if K > 0 else 0.0
    if print_rocpr is True:
        #print(auc_fold)#, sum(np.array(auc_fold))/int(K))
        #print(pr_fold)
        print("************************************************************************")
        print("AVG ", str(K), "AUROC & AUPR", str(avg_auc) + "(" + str(std_auc) + ")",
              str(avg_aupr), "(", str(std_aupr), ")")
        print("************************************************************************")

    # pred_array, auc = classifier_test(df_test[colFeats], 
    #                       df_test["prediction"], clf, scaler)
    # print("TEST AUC on standalone data = ", auc[0])
    # print("individual methods: ", individual_method(df_test))
    return [clf, scaler, avg_auc, std_auc, avg_aupr, std_aupr]



def sim_load_dataset(edge_weights_file):
    # In[6]:
    df = pd.read_csv(edge_weights_file, sep=",")
    if "pcc" in df.columns:
        del df["pcc"]
    if "irafnet" in df.columns:
        del df["irafnet"]
    # del df["grnboost"]
    # for c in df.columns:
    #     print(c, df[c].value_counts())
    df["prediction"].value_counts()
    #
    allOnes = df[df["prediction"]==1]
    allZeros = df[df["prediction"]==0]
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
                            colFeats, output_image, method):
    sim_results = {}
    nfeat = str(len(colFeats))
    _,_, auroc, sroc, aupr, spr = pandas_classifier(df_train, df_test, 
            colFeats, output_image, method=method, ind_fold_auc=False)
    sim_results["ENGRAIN"] = [auroc, aupr]
    sim_results["STD_ENGRAIN"] = [sroc, spr]
    ind_rpr = individual_method(df_test, colFeats, rank_avg=True)
    sim_results.update(ind_rpr)
    sim_results["ScaleSum-"+nfeat] = runScaleSum(allOnes, allZeros, colFeats)
    sim_results["ScaleLSum-"+nfeat] = runScaleLSum(allOnes, allZeros, colFeats)
    print(sim_results)
    return sim_results

def sim_analyse(df_train, df_test, allOnes, allZeros,
                output_image_pfx, method):
    colFeats = [c for c in df_train.columns if c not in ["prediction", "edge"]]
    output_image = output_image_pfx + "-full.png"
    sim_results = sim_learn_classify_test(df_train, df_test, allOnes, allZeros,
                                          colFeats, output_image, method)
    if "genie3" in df_train.columns:
       colFeats7 = ["clr", "grnboost", "aracne", "mrnet", "tinge", "wgcna", "genie3"]
       output_image = output_image_pfx + "-top7.png"
       sim_results7 = sim_learn_classify_test(df_train, df_test, allOnes, allZeros,
                                              colFeats7, output_image, method)
       sim_results7 = {"Top7-"+x:y for x,y in sim_results7.items()}
       sim_results.update(sim_results7)
    else:
       colFeats6 = ["clr", "grnboost", "aracne", "mrnet", "tinge", "wgcna"]
       output_image = output_image_pfx + "-top6.png"
       sim_results6 = sim_learn_classify_test(df_train, df_test, allOnes, allZeros,
                                              colFeats6, output_image, method)
       sim_results6 = {"Top6-"+x:y for x,y in sim_results6.items()}
       sim_results.update(sim_results6)
    return pd.DataFrame.from_dict(sim_results, orient="index",
                columns=["AUROC", "AUPR"])

def sim_load_v5_dataset():
    return sim_load_dataset("data/yeast-edge-weights-v5.csv.gz")

def sim_load_v4_dataset():
    return sim_load_dataset("data/yeast-edge-weights-v4.csv.gz")

def sim_load_v3_dataset():
    return sim_load_dataset("data/yeast-edge-weights-v3.csv.gz")

def sim_analysis_data(dataset_version, output_file, method):
    img_pfx = dataset_version + "-analysis-" + method
    if dataset_version == "v5":
        df_train, df_test, allOnes, allZeros = sim_load_v5_dataset()
        print("Running analysis for Simlated dataset v5")
        dfx = sim_analyse(df_train, df_test, allOnes, allZeros,
                          img_pfx, method)
        print(dfx)
    elif dataset_version == "v4":
        df_train, df_test, allOnes, allZeros = sim_load_v4_dataset()
        print("Running analysis for Simlated dataset v4")
        dfx = sim_analyse(df_train, df_test, allOnes, allZeros, 
                          img_pfx, method)
        print(dfx)
    elif dataset_version == "v3":
        df_train, df_test, allOnes, allZeros = sim_load_v3_dataset()
        print("Running analysis for Simlated dataset v3")
        dfx = sim_analyse(df_train, df_test, allOnes, allZeros,
                          img_pfx, method)
        print(dfx)
    if output_file is None:
        output_file = img_pfx + ".csv"
    if dataset_version in ["v3", "v4", "v5"]:
        dfx.to_csv(output_file)
 

def sim_2k_analysis(method):
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
    out_file = "sim_2k_engrain_" + method + ".csv"
    df_2k = pd.DataFrame()
    for kx, fy in files_2k.items():
        df_train, df_test, allOnes, allZeros = sim_load_dataset(fy)
        print("Loaded dataset : ", kx, fy)
        img_pfx = "m" + kx + "-2kanalysis-" + method
        dfx = sim_analyse(df_train, df_test, allOnes, allZeros,
                          img_pfx, method)
        df_2k["AUROC-" + kx] = dfx["AUROC"]
        df_2k["AUPR-" + kx] = dfx["AUPR"]
    df_2k.to_csv(out_file)

def sim_analysis(options_file, output_file, method):
    with open(options_file) as f:
        jsx = json.load(f)
    out_img_pfx = jsx["OUT_PREFIX"]
    sim_df = pd.DataFrame()
    for kx, fy in jsx["INPUT_FILES"].items():
        df_train, df_test, allOnes, allZeros = sim_load_dataset(fy)
        print("Loaded dataset : ", kx, fy)
        dfx = sim_analyse(df_train, df_test, allOnes, allZeros,
                          out_img_pfx + kx, method)
        sim_df["AUROC-" + kx] = dfx["AUROC"]
        sim_df["AUPR-" + kx] = dfx["AUPR"]
    sim_df.to_csv(output_file)


def engrain_ensemble_train(train_files, colFeats, output_image, method,
                           pos_ratio=0.5, neg_ratio=0.5, 
                           scale_pos_weight=None,
                           random_seed=5):
    # split test and train
    if random_seed is not None:
        random.seed(random_seed)
    # Ones
    #tisOnes  = pd.read_csv(train_dir+'/all-positives.csv', sep=',')
    tisOnes  = pd.read_csv(train_files["positives"], sep=',')
    tisOnes['prediction'] = 1
    print(tisOnes.shape)
    nrow1 = int(tisOnes.shape[0])
    nselect1 = int(nrow1*pos_ratio)
    lselect1 = random.sample(list(range(nrow1)), nselect1)
    # Zeros
    tisZeros = pd.read_csv(train_files["negatives"], sep=',')
    tisZeros['prediction'] = 0
    nrow2 = int(tisZeros.shape[0])
    nselect2 = int(nrow2*neg_ratio)
    lselect2 = random.sample(list(range(nrow2)), nselect2)
    # 
    df_tis= pd.concat([tisOnes.iloc[lselect1, ], tisZeros.iloc[lselect2,]], ignore_index=True)
    df_tis = df_tis.fillna(0)
    df_excl = pd.concat([tisOnes.drop(lselect1, axis=0), tisZeros.drop(lselect2, axis=0)], ignore_index=True)
    trained_params_tissue = pandas_classifier(df_tis, df_excl, colFeats,
                                              output_image, method=method, 
                                              scale_pos_weight=scale_pos_weight)
    return trained_params_tissue, df_tis, df_excl



def stacked_ensemble_train(train_files, tissue_types_train,
                           colFeats, output_image, method, df_test=None, 
                           fold_auc=False):
    # training
    df_tis_train = pd.DataFrame([])
    for i, t in enumerate(tissue_types_train):
        #print(train_dir, i, t)
        tisOnes  =  pd.read_csv(train_files["positives"][t], sep=",")
        #  pd.read_csv(train_dir+'/'+t+'-positives.csv', sep=',')
        tisOnes['prediction'] = 1
        tisZeros = pd.read_csv(train_files["negatives"][t], sep=",")
        #  pd.read_csv(train_dir+'/'+t+'-negatives.csv', sep=',')
        tisZeros['prediction'] = 0
        df_tis= pd.concat([tisOnes, tisZeros], ignore_index=True)
        # # Imputation step with zeros
        df_tis = df_tis.fillna(0)
        df_tis_train = pd.concat([df_tis_train, df_tis], ignore_index=True)
    #print(df_tis_train.head(), colFeats)
    if df_test is None:
       trained_params_tissue = pandas_classifier(df_tis_train, df_tis_train,
                    colFeats, output_image, method=method, ind_fold_auc=fold_auc)
    else:
       trained_params_tissue = pandas_classifier(df_tis_train, df_test,
               colFeats, output_image, method, ind_fold_auc=fold_auc)
    return trained_params_tissue



def engrain_predict(train_files, network_file, colFeats, output_image, method,
                    pos_ratio=0.5, neg_ratio=0.5, scale_pos_weight=None,
                    random_seed=5):
    a, df_in, df_out =  engrain_ensemble_train(train_files, colFeats, output_image,
                                            method, pos_ratio, neg_ratio, 
                                            scale_pos_weight, random_seed)
    clf, scaler, tauroc, _, taupr, _ = a
    if network_file is not None:
        df_tis  = pd.read_csv(network_file, sep=',')
        df_tis = df_tis.fillna(0)
        print("Loaded Network file : ", network_file)
        pred_array = classifier_prediction(df_tis[colFeats], clf, scaler, None)
        print("Finished prediction for Network file : ", network_file)
        df_tis = df_tis[['edge']+colFeats]
        df_tis['wt'] = pred_array
        df_tis.sort_values(by=['wt'], inplace=True, ascending=False)
        return df_tis, df_in, df_out, tauroc, taupr
    else:
        return None, df_in, df_out, tauroc, taupr

def engrain_ensemble_predict(options_file,  network_file, output_file, method):
    with open(options_file) as jfptr:
        jsx = json.load(jfptr)
    #train_files = {"positives": train_dir + "/all-positives.csv",
    #               "negatgives": train_dir + "/all-negatives.csv"}
    train_dir = jsx["DATA_DIR"]  + "/"
    train_files = {"positives" : train_dir + jsx["POSITIVES_TABLES"],
                   "negatives" : train_dir + jsx["NEGATIVES_TABLES"]}
    colFeats = jsx["FEATURES"]
    output_image = jsx["OUT_PREFIX"] + "-feats.png"
    df_tis, df_in, df_out, tauroc, taupr = engrain_predict(train_files,
                                        network_file, colFeats, output_image, method)
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



def engrain_stacked_ensemble_predict(options_file, network_file, 
                                     output_file, method):
    with open(options_file) as jfptr:
        jsx = json.load(jfptr)
    data_dir = jsx["DATA_DIR"]  + "/"
    tissue_types_train = jsx["TRAIN_SUBSETS"] 
    tissue_types_test = jsx["TEST_SUBSETS"]
    colFeats = jsx["FEATURES"]
    output_image = jsx["OUT_PREFIX"] + "-feats.png"
    pos_files = {x: data_dir + fx for x, fx in jsx["POSITIVES_TABLES"].items()}
    neg_files = {x: data_dir + fx for x, fx in jsx["NEGATIVES_TABLES"].items()}
    train_files = {"positives" : pos_files, "negatives": neg_files}
    print("TRAIN WITH Features:", ",".join(colFeats), end="; ")
    clf, scaler, _, _, _, _ = stacked_ensemble_train(
            train_files, tissue_types_train, colFeats, output_image, method)
    # if network file and output file are available, then generate predictions.
    if network_file is not None and output_file is not None:
        df_tis  = pd.read_csv(network_file, sep=',')
        df_tis = df_tis.fillna(0)
        print("Loaded Network file : ", network_file)
        pred_array = classifier_prediction(df_tis[colFeats], clf, scaler, None)
        print("Finished prediction for Network file : ", network_file)
        df_tis['wt'] = pred_array
        df_tis.to_csv(output_file, sep="\t", index=False)



def ensemble_grid_search_rocpr(options_file, output_file, method):
    with open(options_file) as jfptr:
        jsx = json.load(jfptr)
    train_dir = jsx["DATA_DIR"]  + "/"
    train_files = {"positives" : train_dir + jsx["POSITIVES_TABLES"],
                   "negatives" : train_dir + jsx["NEGATIVES_TABLES"]}
    output_image = jsx["OUT_PREFIX"]  + ".png"
    pos_range = jsx["POS_RANGE"]
    neg_range = jsx["NEG_RANGE"]
    colFeats = jsx["FEATURES"]
    ropt = [(p, q, None) for p in pos_range for q in neg_range]
    for p in jsx["POS_RANGE"]:
        for q in jsx["NEG_RANGE"]:
            for r in  jsx["POS_WEIGHT_RANGE"]:
                ropt.append((float(p), float(q), float(r))) 
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
    ofptr = open(output_file, "w")
    ofptr.write(",".join([str(x) for x in oelts]) + "\n")
    for p,q,r in ropt:
        pos_ratio = float(p)
        neg_ratio = float(q)
        scale_pos_weight = r if r is None else float(r)
        aupr_lst = [0.0 for _ in range(nrounds)]
        auroc_lst = [0.0 for _ in range(nrounds)]
        for i in range(nrounds):
            a, df_in, df_out =  engrain_ensemble_train(train_files, colFeats, 
                                         output_image, method, pos_ratio, neg_ratio, 
                                         scale_pos_weight, random_seed)
            _, _, tauroc, _, taupr, _ = a
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
            ofptr.write(",".join([str(x) for x in oelts]) + "\n")
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
        ofptr.write(",".join([str(x) for x in oelts]) + "\n")
    ofptr.close()

def ensemble_grid_search_tpfp(options_file, network_file, 
                              output_file, method):
    # network and output file1
    #
    with open(options_file) as jfptr:
        jsx = json.load(jfptr)
    train_dir = jsx["DATA_DIR"]  + "/"
    train_files = {"positives" : train_dir + jsx["POSITIVES_TABLES"],
                   "negatives" : train_dir + jsx["NEGATIVES_TABLES"]}
    output_image = jsx["OUT_PREFIX"]  + ".png"
    pos_range = jsx["POS_RANGE"]
    neg_range = jsx["NEG_RANGE"]
    ropt = [(p, q, None) for p in pos_range for q in neg_range]
    for p in jsx["POS_RANGE"]:
        for q in jsx["NEG_RANGE"]:
            for r in  jsx["POS_WEIGHT_RANGE"]:
                ropt.append((float(p), float(q), float(r))) 
    colFeats = jsx["FEATURES"]
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
    ofptr = open(output_file, "w")
    print("\t".join([str(x) for x in oelts]))
    ofptr.write(",".join([str(x) for x in oelts]) + "\n")
    for p,q,r in ropt:
        pos_ratio = float(p)
        neg_ratio = float(q)
        scale_pos_weight = r if r is None else float(r)
        df_tis, df_in, df_out, tauroc, taupr = engrain_predict(train_files,
                                        network_file, colFeats, output_image, 
                                        method,
                                        pos_ratio, neg_ratio, scale_pos_weight,
                                        random_seed)
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
            ofptr.write(",".join([str(x) for x in oelts]) + "\n")
    ofptr.close()



def cross_tissue_ensemble(options_file, output_file, method):
    with open(options_file) as fx:
        jsx = json.load(fx)
    out_img_pfx = jsx["OUT_PREFIX"]
    data_dir = jsx["DATA_DIR"] + "/" # "data/athaliana_raw_filtered/"
    tissue_types = jsx["TEST_SUBSETS"]
    train_types = jsx["TRAIN_SUBSETS"]
    colFeats = jsx["FEATURES"]
    pos_files = {x: data_dir + fx for x, fx in jsx["POSITIVES_TABLES"].items()}
    neg_files = {x: data_dir + fx for x, fx in jsx["NEGATIVES_TABLES"].items()}
    train_files = {"positives" : pos_files, "negatives": neg_files}
    pos_test_tables = {x: pd.read_csv(pos_files[x]) for x in tissue_types}
    neg_test_tables = {x: pd.read_csv(neg_files[x]) for x in tissue_types}
    ##
    ##
    rst_lst = []
    for tissue_types_train in train_types:
        #colFeats = ["clr", "grnboost"]
        output_image = out_img_pfx + "-" + "-".join(tissue_types_train)  + ".png"
        print("TRAIN SET:", tissue_types_train, ";OUTPUT IMAGE:", output_image, end="; ")
        clf, scaler, _, _, _, _ = stacked_ensemble_train(train_files, 
                     tissue_types_train, colFeats, output_image, method)
        tissue_types_test = list(set(tissue_types) - set(tissue_types_train))
        ensemble_dct = { x: "Tr" for x in tissue_types_train}
        # testing
        # predicting using the trained params
        for tx in tissue_types_test:
            pos_df = pos_test_tables[tx]
            pos_df.fillna(0, inplace=True)
            pos_df["prediction"] = 1
            neg_df = neg_test_tables[tx]
            neg_df.fillna(0, inplace=True)
            neg_df["prediction"] = 0
            df_test = pd.concat([pos_df, neg_df], ignore_index=True)
            # 
            print("TISSUE:", tx, ";COLS:", len(df_test.columns), 
                    ";FEATS:", len(colFeats), ";SHAPE:",
                    df_test.shape, ";NAs:", df_test["prediction"].isna().sum(), end=" ")
            pred_array, auc = classifier_test(df_test[colFeats], df_test["prediction"],
                                              clf, scaler, None, method=method)
            print("ENSEMBLE", tx, "AUC:", auc[0], ";AUPR:", auc[1], end=" ")
            #print("individual methods on arabidopsis: [auroc, aupr]")
            ind_results = individual_method(df_test, colFeats,
                                            rank_avg=False, print_rocpr=False)
            ensemble_compare = True
            for r in ind_results:
                #print(r, ind_results[r]<auc[0])
                ensemble_compare &= ind_results[r][0]<auc[0]
            print("BETTTER ?", "Y" if ensemble_compare else "N")
            ensemble_dct[tx] = "Y" if ensemble_compare else "N"
        rst_lst.append(ensemble_dct)
    pd.DataFrame(rst_lst).to_csv(output_file)

def feature_union_net(df_tis, union_feats):
    data_dct = {'edge' : df_tis.edge}
    for x in union_feats:
        data_dct[x["name"]] = df_tis[x["feats"]].max(axis=1)
    data_dct['prediction'] = df_tis.prediction
    df = pd.DataFrame(data=data_dct)
    return df

def avgrank_stats(df, colFeats):
    df_tis = df[colFeats + ["prediction"]]
    df_tis.wt = 1/df_tis[colFeats].rank(ascending=False).mean(axis=1)
    aupr = metrics.average_precision_score(df_tis.prediction, df_tis.wt)
    auroc = metrics.roc_auc_score(df_tis.prediction, df_tis.wt)
    return auroc, aupr

def rankavg_nets(train_files, colFeats, output_file):
    #tisOnes  = pd.read_csv(train_dir+'/all-positives.csv', sep=',')
    tisOnes  = pd.read_csv(train_files["positives"], sep=',')
    tisOnes['prediction'] = 1
    #print(tisOnes.shape)
    # Zeros
    #tisZeros = pd.read_csv(train_dir+'/all-negatives.csv', sep=',')
    tisZeros = pd.read_csv(train_files["negatives"], sep=',')
    tisZeros['prediction'] = 0
    df_tis= pd.concat([tisOnes, tisZeros], ignore_index=True)
    df_tis = df_tis.fillna(0)
    avg_auroc, avg_aupr = avgrank_stats(df_tis, colFeats)
    ss_auroc, ss_aupr = scalesum_stats(df_tis, colFeats)
    methods = ["aracne", "clr", "grnboost", "mrnet", "tinge", "wgcna"]
    labels = ["ARACNe-AP", "CLR", "GRNBoost", "MRNET", "TINGe", "WGCNA"]
    with open(output_file, 'w') as ofx:
        ofx.write(",".join(["FEATS", "NFEATS", "AUROC", "AUPR"])+"\n")
        ofx.write(",".join([str(x) for x in ["all-avergrank", len(colFeats), 
                                             avg_auroc, avg_aupr]])+"\n")
        ofx.write(",".join([str(x) for x in ["all-scalelsum", len(colFeats),
                                             ss_auroc, ss_aupr]])+"\n")
        for x,z in zip(methods, labels):
            xfeats = [y for y in colFeats if y.endswith(x)]
            auroc, aupr = avgrank_stats(df_tis, xfeats)
            auc_line =  ",".join(str(r) for r in [z+ " Rank Avg.", len(xfeats),
                                  round(auroc, 4), round(aupr, 4)])
            ofx.write(auc_line+"\n")
        for x,z in zip(methods, labels):
            xfeats = [y for y in colFeats if y.endswith(x)]
            auroc, aupr = scalesum_stats(df_tis, xfeats)
            auc_line = ",".join(str(r) for r in [z+ "\\textit{ScaleSum}", 
                           len(xfeats), round(auroc, 4), round(aupr, 4)])
            ofx.write(auc_line+"\n")
 

def rankavg_of_union_nets(train_files, union_feats, output_file):
    #tisOnes  = pd.read_csv(train_dir+'/all-positives.csv', sep=',')
    tisOnes  = pd.read_csv(train_files["positives"], sep=',')
    tisOnes['prediction'] = 1
    #print(tisOnes.shape)
    # Zeros
    #tisZeros = pd.read_csv(train_dir+'/all-negatives.csv', sep=',')
    tisZeros = pd.read_csv(train_files["negatives"], sep=',')
    tisZeros['prediction'] = 0
    df_tis= pd.concat([tisOnes, tisZeros], ignore_index=True)
    df_tis = df_tis.fillna(0)
    df_union = feature_union_net(df_tis, union_feats)
    df_union.wt = 1/df_union.rank(ascending=False).mean(axis=1)
    aupr = metrics.average_precision_score(df_union.prediction, df_union.wt)
    auroc = metrics.roc_auc_score(df_union.prediction, df_union.wt)
    ss_methods = [x["name"] for x in union_feats]
    ss_auroc, ss_aupr = scalesum_stats(df_union, ss_methods)
    with open(output_file, "w") as ofx:
        ofx.write(",".join(["FEATS", "NFEATS", "AUROC", "AUPR"])+"\n")
        ofx.write(",".join([str(x) for x in ["Union-Max", 
            sum(len(x) for x in union_feats), auroc, aupr]])+"\n")
        ofx.write(",".join([str(x) for x in ["all-scalelsum", 
            len(ss_methods), ss_auroc, ss_aupr]])+"\n")
        for x in ss_methods:
            aupr = metrics.average_precision_score(df_union.prediction, df_union[x])
            auroc = metrics.roc_auc_score(df_union.prediction, df_union[x])
            ofx.write(",".join([str(y) for y in [x, 20, auroc, aupr]])+"\n")


def analyse_full_rankavg(options_file, output_file):
    with open(options_file) as fx:
        jsx = json.load(fx)
    train_dir = jsx["DATA_DIR"]  + "/"
    train_files = {"positives" : train_dir + jsx["POSITIVES_TABLES"],
                   "negatives" : train_dir + jsx["NEGATIVES_TABLES"]}
    #output_image = jsx["OUT_PREFIX"]  + ".png"
    ravg_feats = jsx["COL_FEATS"]
    rankavg_nets(train_files, ravg_feats, output_file)
 

def analyse_rankavg_of_union(options_file, output_file):
    with open(options_file) as fx:
        jsx = json.load(fx)
    train_dir = jsx["DATA_DIR"]  + "/"
    train_files = {"positives" : train_dir + jsx["POSITIVES_TABLES"],
                   "negatives" : train_dir + jsx["NEGATIVES_TABLES"]}
    #output_image = jsx["OUT_PREFIX"]  + ".png"
    union_feats = jsx["UNION_FEATS"]
    rankavg_of_union_nets(train_files, union_feats, output_file)


if __name__ == "__main__":
    PROG_DESC = """Train with a subset of network and predict Ensemble """
    PARSER = argparse.ArgumentParser(description=PROG_DESC)
    SUBPARSERS = PARSER.add_subparsers(dest="sub_cmd")
    # 
    # "sim" command
    # Runs Ensemble for the simulated datasets
    PARSER_SIM = SUBPARSERS.add_parser("sim", help="Run Simulated Datasets")
    PARSER_SIM.add_argument("options_file", help="""JSON File w. Input Options.
            Apart from an json input file with the options, this argument can 
            accept one of the following values: v3, v4, v5 and 2k.
            Arguments "v3", "v4", "v5" runs EnGRaiN for the v3,v4,v5 versions 
            of the dataset respectively.
            Option "2k" runs the simulated datsets of 2000 genes with 
            250,500,750,1000,1250,1500,1750 & 2000 observations.""")
    PARSER_SIM.add_argument("-o", "--output_file", default=None,
                            help="""Output File""")
      ## Supervison method
    PARSER_SIM.add_argument("-m", "--super_method", default='xgb',
                 choices=['svc', 'rf', 'mlp', 'sgd', 'xgb'],
                 help="""Should be one of 'xgb', 'rf', 'svc', 'mlp', 'sgd'""")
    # ravgu: Rank average of union networks 
    PARSER_RVGU = SUBPARSERS.add_parser("ravgu", 
            help="Run Rank Avg.&ScaleSum of union networks for A. thaliana Datasets")
    PARSER_RVGU.add_argument("options_file", help="""JSON File w. Input Options""")
    PARSER_RVGU.add_argument("-o", "--output_file", default=None,
                             help="""Output File""")
    # ravg: Rank average of networks 
    PARSER_RVG = SUBPARSERS.add_parser("ravg", 
            help="Run Rank Avg.&ScaleSum of networks for A. thaliana Datasets")
    PARSER_RVG.add_argument("options_file", help="""JSON File w. Input Options""")
    PARSER_RVG.add_argument("-o", "--output_file", default=None,
                             help="""Output File""")
    # "xtissue" command:  Cross tissue command 
    PARSER_XTS = SUBPARSERS.add_parser("xtissue", 
            help="Run Cross-tissue comparisions A. thaliana Datasets")
    PARSER_XTS.add_argument("options_file", help="""JSON File w. Input Options""")
    PARSER_XTS.add_argument("-o", "--output_file", default=None,
                            help="""Output File""")
      ## Supervison method
    PARSER_XTS.add_argument("-m", "--super_method", default='xgb',
                 choices=['svc', 'rf', 'mlp', 'sgd', 'xgb'],
                 help="""Should be one of 'xgb', 'rf', 'svc', 'mlp', 'sgd'""")
    #
    # The "predicts" command: Stacked predict
    PARSER_PRED = SUBPARSERS.add_parser("pred_stk", 
                help="Stacked Predictions for A. thaliana Networks")
    PARSER_PRED.add_argument("options_file", help="""JASON File w. Input Options""")
    PARSER_PRED.add_argument("-n", "--network_file",
                        default="edge_networks/union-all-networks.csv",
                        help="""network build from a reverse engineering methods
                                (network is a csv file)""")
      ## Supervison method
    PARSER_PRED.add_argument("-m", "--super_method", default='xgb',
                 choices=['svc', 'rf', 'mlp', 'sgd', 'xgb'],
                 help="""Should be one of 'xgb', 'rf', 'svc', 'mlp', 'sgd'""")
    PARSER_PRED.add_argument("-o", "--output_file", default=None,
                         help="""Output File""")
    #
    #
    #
    PARSER_ENS = SUBPARSERS.add_parser("pred_ens", 
                help="Ensemble Predictions for A. thaliana Networks")
    PARSER_ENS.add_argument("options_file", help="""JASON File w. Input Options""")
    PARSER_ENS.add_argument("-n", "--network_file",
                        default="edge_networks/union-all-networks.csv",
                        help="""network build from a reverse engineering methods
                                (network is a csv file)""")
    PARSER_ENS.add_argument("-o", "--output_file", default=None,
                         help="""Output File""")
      ## Supervison method
    PARSER_ENS.add_argument("-m", "--super_method", default='xgb',
                 choices=['svc', 'rf', 'mlp', 'sgd', 'xgb'],
                 help="""Should be one of 'xgb', 'rf', 'svc', 'mlp', 'sgd'""")
    # "grid_rocpr" command: Grid Search with Arabidopsis and eval ROC/PR
    PARSER_GDS = SUBPARSERS.add_parser("grids_rocpr", 
            help="Run Grid Search with XGBoost params for A. thaliana Datasets")
    PARSER_GDS.add_argument("options_file", help="""JSON File w. Input Options""")
    PARSER_GDS.add_argument("-o", "--output_file", default=None,
                            help="""Output File""")
       ## Supervison method
    PARSER_GDS.add_argument("-m", "--super_method", default='xgb',
                 choices=['svc', 'rf', 'mlp', 'sgd', 'xgb'],
                 help="""Should be one of 'xgb', 'rf', 'svc', 'mlp', 'sgd'""")
    # "grid_tpfp" command: Grid Search with Arabidopsis and eval TP/FP
    PARSER_GDSTFP = SUBPARSERS.add_parser("grids_tpfp", 
            help="Run Grid Search with XGBoost params for A. thaliana Datasets")
    PARSER_GDSTFP.add_argument("options_file", help="""JSON File w. Input Options""")
    PARSER_GDSTFP.add_argument("-n", "--network_file",
                       default="edge_networks/union-all-networks.csv",
                       help="""network build from a reverse engineering methods
                                (network is a csv file)""")
    PARSER_GDSTFP.add_argument("-o", "--output_file", default=None,
                               help="""Output File""")
       ## Supervison method
    PARSER_GDSTFP.add_argument("-m", "--super_method", default='xgb',
                 choices=['svc', 'rf', 'mlp', 'sgd', 'xgb'],
                 help="""Should be one of 'xgb', 'rf', 'svc', 'mlp', 'sgd'""")
 

    # Prase args
    ARGS = PARSER.parse_args()
    print(ARGS) 
    run_type = ARGS.sub_cmd
    print("Run Type : ",  run_type)
    print(" -> Option File : ",  ARGS.options_file)
    print(" -> Output File : ",  ARGS.output_file)
    if run_type == "sim":
        if ARGS.options_file in ["v3", "v4", "v5"]:
            sim_analysis_data(ARGS.options_file, ARGS.output_file, ARGS.super_method)
        elif ARGS.options_file == "2k":
            sim_2k_analysis(ARGS.super_method)
        else:
            sim_analysis(ARGS.options_file, ARGS.output_file, ARGS.super_method)
    elif run_type == "xtissue":
        cross_tissue_ensemble(ARGS.options_file, ARGS.output_file, 
                              ARGS.super_method) 
    elif run_type == "ravgu":
        analyse_rankavg_of_union(ARGS.options_file, ARGS.output_file) 
    elif run_type == "ravg":
        analyse_full_rankavg(ARGS.options_file, ARGS.output_file)
    elif run_type == "pred_stk":
        print(" -> Network File : ",  ARGS.network_file)
        engrain_stacked_ensemble_predict(ARGS.options_file, ARGS.network_file,
                                         ARGS.output_file, ARGS.super_method)
    elif run_type == "pred_ens":
        print(" -> Network File : ",  ARGS.network_file)
        engrain_ensemble_predict(ARGS.options_file, ARGS.network_file,
                                 ARGS.output_file, ARGS.super_method)
    elif run_type == "grids_rocpr":
        ensemble_grid_search_rocpr(ARGS.options_file, ARGS.output_file, 
                                   ARGS.super_method)
    elif run_type == "grids_tpfp":
        ensemble_grid_search_tpfp(ARGS.options_file, ARGS.network_file, 
                                  ARGS.output_file, ARGS.super_method)
    else:
        print("Invalid Command!!!")
 
