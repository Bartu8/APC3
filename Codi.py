# -*- coding: utf-8 -*-
import random as random
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import svm, datasets
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_recall_curve, average_precision_score, roc_curve, auc

def load_dataset(path):
    dataset = pd.read_csv(path, header=0, delimiter=',', names=["ID", "Game", "State", "Hours", "0"])
    return dataset

def set_dataset(dataset):
    #Per a cada jugador, ja que entre jugadors si hi haura repetits
    datasetfinal=pd.DataFrame()
    for ID in dataset["ID"].unique():
        #Juntem les columnes State i Hours
        #Separem del dataset entre State==play
        datasetID=dataset.loc[dataset["ID"]==ID]
        played=datasetID.loc[datasetID["State"]=='play']
        played=played.drop(['State'], axis=1)
        
        #Separem del dataset entre State==pucharse i eliminem els jocs que ja siguin en played
        purchased=datasetID.loc[datasetID["State"]=='purchase']
        purchased=pd.merge(purchased["Game"],played["Game"], indicator=True, how='outer').query('_merge=="left_only"').drop(['_merge'],axis=1)
        purchased=purchased.assign(ID=ID)
        columns_titles = ["ID","Game"]
        purchased=purchased.reindex(columns=columns_titles)
        purchased=purchased.assign(Hours=0)
      
        #Juntem played i pucharsed
        datasetID=pd.concat([played, purchased])
    
        PriceList=[]
        ValueList=[]
        LogList=[]
        WorthList=[]
        threshold=[0.02, 0.2]#1€/h -> 0.2, 2€/hora -> 0.02
        
        #Afegim el preu, el value es Preu/Hores, i li assignem una categoria entre 0,1,2
        for index, row in datasetID.iterrows():
            price=preusJocs[row["Game"]]
            PriceList.append(price)
            
            if row["Hours"] == 0:
                value=999
            else:
                value=price/row["Hours"]
                
            ValueList.append(value)
            value=(10**-value)*2
            LogList.append(value)
            
            if value < threshold[0]:
                WorthList.append(0)
            elif value < threshold[1]:
                WorthList.append(1)
            else:
                WorthList.append(2)
                
                
        datasetID["Price"]=PriceList
        datasetID["Value"]=ValueList
        datasetID["Log"]=LogList
        datasetID["Worth"]=WorthList
        datasetfinal=pd.concat([datasetfinal, datasetID])
    return datasetfinal

def draw_scatterheatmap(dataset):
    rangeY=2
    for i in range(1,len(dataset.columns)-1):
        atribut1=dataset.columns[i]
        for y in range(rangeY,len(dataset.columns)):
            atribut2=dataset.columns[y]
            plt.figure()
            plt.scatter(dataset[atribut1], dataset[atribut2])
            plt.xlabel(atribut1)
            plt.savefig('AnalisiDades/scatter ' + atribut1 + "-"+ atribut2)
        rangeY+=1
        
    correlacio = dataset.corr()
    plt.figure()
    sns.heatmap(correlacio, annot=True, linewidths=.5)
    plt.savefig('AnalisiDades/Heatmap')
    
def make_meshgrid(x, y, h=1):
        """Create a mesh of points to plot in
        Parameters
        ----------
        x: data to base x-axis meshgrid on
        y: data to base y-axis meshgrid on
        h: stepsize for meshgrid, optional
        Returns
        -------
        xx, yy : ndarray
        """
        x_min, x_max = x.min() - 1, x.max() + 1
        y_min, y_max = y.min() - 1, y.max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                              np.arange(y_min, y_max, h))
        return xx, yy


def plot_contours(ax, clf, xx, yy, **params):
    """Plot the decision boundaries for a classifier.
    Parameters
    ----------
    ax: matplotlib axes object
    clf: a classifier
    xx: meshgrid ndarray
    yy: meshgrid ndarray
    params: dictionary of params to pass to contourf, optional
    """
    a = xx.ravel()
    b = yy.ravel()
    c = np.c_[a, b]
    d = clf.predict(c)
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out

def show_C_effect(X, y, C=1.0, gamma=0.7, degree=3):

    # we create an instance of SVM and fit out data. We do not scale our
    # data since we want to plot the support vectors
    # title for the plots
    titles = ('SVC with linear kernel',
              'LinearSVC (linear kernel)',
              'SVC with RBF kernel',
              'SVC with polynomial (degree 3) kernel')

    #C = 1.0  # SVM regularization parameter
    models = (svm.SVC(),
              svm.LinearSVC(C=C, max_iter=1000000),
              svm.SVC(kernel='rbf', gamma=gamma, C=C),
              svm.SVC(kernel='poly', degree=degree, gamma='auto', C=C))
    models = (clf.fit(X, y) for clf in models)

    plt.close('all')
    fig, sub = plt.subplots(2, 2, figsize=(14,9))
    plt.subplots_adjust(wspace=0.4, hspace=0.4)


    X0 = X.loc[:, "Hours"]
    X1 = X.loc[:, "Price"]
    xx, yy = make_meshgrid(X0, X1)
    for clf, title, ax in zip(models, titles, sub.flatten()):
        plot_contours(ax, clf, xx, yy,
                      cmap=plt.cm.coolwarm, alpha=0.8)
        ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xlabel('Sepal length')
        ax.set_ylabel('Sepal width')
        ax.set_xticks(())
        ax.set_yticks(())
        ax.set_title(title)
    plt.savefig("AnalisiResultats/Scatter Meshgrid 2")
    
def classificacio(x, y, partes):
    for parte in partes:
        x_t, x_v, y_t, y_v = train_test_split(x, y, train_size=parte)
        
        # Creem el regresor logístic
        logireg = LogisticRegression()
        
        # l'entrenem
        logireg.fit(x_t, y_t)
        print("Correct classification Logistic ",parte, "% of the data: ", logireg.score(x_v, y_v))
        
        # Creem la maquina de vectors
        svc = svm.SVC(probability=True)
        
        # l'entrenem
        svc.fit(x_t, y_t)
        probs = svc.predict(x_v)
        print("Correct classification SVM      ",parte, "% of the data: ", svc.score(x_v, y_v))

def PRROC(lista_modelos, text_modelos, part):
    for a, model in enumerate(lista_modelos):
        x_t, x_v, y_t, y_v = train_test_split(x, y, train_size=part)
        model.fit(x_t, y_t)
        probs = model.predict_proba(x_v)
    
        precision = {}
        recall = {}
        average_precision = {}
        plt.figure()
        for i in range(len(Valor)):
            precision[i], recall[i], _ = precision_recall_curve(y_v == i, probs[:, i])
            average_precision[i] = average_precision_score(y_v == i, probs[:, i])
            #plt.figure()
            #label='Precision-recall curve of class {0} (area = {1:0.2f})' ''.format(i, average_precision[i])
            plt.plot(recall[i], precision[i])
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.legend(loc="upper right")
        plt.savefig("AnalisiResultats/Precission Recall Curve " + text_modelos[a])
        
        fpr = {}
        tpr = {}
        roc_auc = {}
        for i in range(len(Valor)):
            fpr[i], tpr[i], _ = roc_curve(y_v == i, probs[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
    
        # Compute micro-average ROC curve and ROC area
        # Plot ROC curve
        plt.figure()
        for i in range(len(Valor)):
            plt.plot(fpr[i], tpr[i], label='ROC curve of class {0} (area = {1:0.2f})' ''.format(i, roc_auc[i]))
        plt.savefig("AnalisiResultats/ROC " + text_modelos[a])

        
###############################################
       
Valor={0:"Worthless", 1:"Fair enough worth", 2:"So worth"}

dataset = load_dataset('BBDD/steam-200k.csv')
dataset = dataset.drop(['0'], axis=1)
dataset = dataset.loc[:10000, :]

#Definim un preu fix per a cada joc
preusJocs = {}
for game in dataset["Game"].unique():
    preusJocs[game]=random.randrange(10, 80, 5)
    
    
dataset=set_dataset(dataset)

draw_scatterheatmap(dataset)

x=dataset[["Hours", "Price"]]
y=dataset[["Worth"]]
y=y.values.ravel()
partes=[0.1, 0.2, 0.5]
classificacio(x, y, partes)


lista_modelos = [LogisticRegression(), svm.SVC(probability=True)]
text_modelos = ["Logistic Regression", "SVM"]
PRROC(lista_modelos, text_modelos, partes[2])

show_C_effect(x, y, C=10)