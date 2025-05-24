import os
import copy
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn import preprocessing
from sklearn import neighbors
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from time import time
from sklearn.naive_bayes import MultinomialNB
from sklearn import tree
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn import metrics
from sklearn.preprocessing import label_binarize
from sklearn.metrics import precision_recall_fscore_support
import numpy as np
import statistics

def RF():
    clf = svm.SVC()
    return clf

neighbors.KNeighborsClassifier()


def getData_3(x):
    from sklearn.preprocessing import MinMaxScaler  # 确保导入
    fPath = 'D:\E盘\数据\microarray data\SRBCT.csv'
    dataMatrix = np.array(pd.read_csv(fPath, header=None, skiprows=1))
    rowNum, colNum = dataMatrix.shape[0], dataMatrix.shape[1]
    sampleData = []
    sampleClass = []

    double = []
    temp = []
    a = np.array(x)

    for i in range(0, rowNum):
        for j in a:
            temp.append(dataMatrix[i, j])
        temp.append(dataMatrix[i, -1])
        double.append(temp)
        temp = []
    double = np.array(double)

    for i in range(0, rowNum):
        tempList = list(double[i, :])
        sampleClass.append(tempList[-1])
        sampleData.append(tempList[:-1])
    sampleM = np.array(sampleData)
    classM = np.array(sampleClass)

    # 🔴 添加归一化
    scaler = MinMaxScaler()
    sampleM = scaler.fit_transform(sampleM)

    skf = StratifiedKFold(n_splits=5)
    setDict = {}
    count = 1

    for trainI, testI in skf.split(sampleM, classM):
        trainSTemp, trainCTemp = [], []
        testSTemp, testCTemp = [], []

        for t1 in trainI:
            trainSTemp.append(list(sampleM[t1, :]))
            trainCTemp.append(classM[t1])
        setDict[str(count) + 'train'] = np.array(trainSTemp)
        setDict[str(count) + 'trainclass'] = np.array(trainCTemp)

        for t2 in testI:
            testSTemp.append(list(sampleM[t2, :]))
            testCTemp.append(classM[t2])
        setDict[str(count) + 'test'] = np.array(testSTemp)
        setDict[str(count) + 'testclass'] = np.array(testCTemp)

        count += 1
    return setDict



def getRecognitionRate(testPre, testClass):
    testNum = len(testPre)
    rightNum = 0
    for i in range(0, testNum):
        if testClass[i] == testPre[i]:
            rightNum += 1
    return float(rightNum) / float(testNum)


def cal(x, setNums):
    clf_RF = RF()
    RF_rate = 0.0
    AUC = 0.0
    recall = 0.0
    f = 0.0
    precision = 0.0
    #setDict = getData_3()
    #setNums = len(setDict.keys()) / 4
    for i in range(1, int(setNums + 1)):
        X_train = x[str(i) + 'train']
        y_train = x[str(i) + 'trainclass']
        X_test = x[str(i) + 'test']
        y_test = x[str(i) + 'testclass']
        clf_RF.fit(X_train, y_train)
        RF_rate += getRecognitionRate(clf_RF.predict(X_test), y_test)
        # y_train = y_train
        # y_test = y_test
        # class_names = np.unique(y_train)
        # y_binarize = label_binarize(y_test, classes=class_names)
        # y_fit = label_binarize(clf_RF.predict(X_test), classes=class_names)
        # fpr, tpr, _ = roc_curve(y_binarize.ravel(), y_fit.ravel())
        # AUC += metrics.auc(fpr, tpr)
        # C,D,E,F = precision_recall_fscore_support(y_test, clf_RF.predict(X_test))
        # precision += np.mean(C)
        # recall += np.mean(D)
        # f += np.mean(E)
    A = RF_rate / float(setNums)
    # B = AUC / float(setNums)
    # C = recall / float(setNums)
    # D = f / float(setNums)
    # E = precision / float(setNums)
    return (A)


if __name__ == "__main__":
        z = []
        y = [245, 1463, 1953, 1388, 544, 1318, 565, 35, 1612, 1326, 364, 1707, 138, 1644, 1329, 1073, 347, 28, 1990, 367, 693, 118, 429, 743, 408, 2116, 802, 1069, 2252, 819, 1496, 1941, 51, 606, 1226, 532, 1297, 1517, 1488, 1700, 1485, 1713, 406, 1771, 778, 1489, 972, 1291, 1836, 1830, 1760, 256, 335, 827, 750, 1088, 1872, 1916, 74, 31, 181, 1607, 1439, 831, 2299, 1905, 633, 1875, 377, 2301, 510, 1670, 339,1699, 730, 243, 2302, 1225, 372, 1358, 635, 263, 1979, 1523, 476, 481, 1072, 373, 2098, 1878, 509, 603, 1324, 788, 1119, 137, 1625, 503, 170, 0, 787, 1669, 2020, 811, 2071, 1725, 1727, 1316, 1877, 483,407, 1690, 978, 1155, 2161, 649, 1142, 1755, 387, 1319, 1854, 1093, 1827, 259, 1521, 79, 846, 836, 1285, 1092, 468, 1649, 1404, 26, 1048, 1045, 1480, 875, 88, 1708, 1733, 1220, 705, 520, 837, 890, 2222,1866, 103, 1020, 83, 1913, 670, 382, 1978, 712, 850, 2229, 695, 1911, 1245, 1770, 411, 63, 1038, 1516, 450, 1209, 950, 341, 96, 1524, 1022, 106, 1728, 1377, 773, 464, 317, 401, 1121, 799, 946, 2019, 823, 1633, 1033, 1261, 309, 2008, 1950, 2057, 108, 32, 2287, 353, 2235, 1518, 1232]
        print(y[:3])
        for a in range(199):
            x = y[:a+1]
            setDict = getData_3(x)
            setNums = len(setDict.keys()) / 4
            w = cal(setDict, setNums)
            print(w)
            z.append(w)
        print(z)
