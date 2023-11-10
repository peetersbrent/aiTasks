import streamlit as st
import pandas as pd
from matplotlib import pyplot as plt

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn import metrics

import pydotplus
from ucimlrepo import fetch_ucirepo 

from sklearn.multiclass import OneVsOneClassifier
from sklearn.svm import LinearSVC

from sklearn.neighbors import (NeighborhoodComponentsAnalysis,
KNeighborsClassifier)
from sklearn.pipeline import Pipeline

  
# fetch dataset 
glass_identification = fetch_ucirepo(id=42) 
  
# data (as pandas dataframes) 
X = glass_identification.data.features
y = glass_identification.data.targets 

feature_cols = ['RI','Na','Mg','Al','Si','K','Ca','Ba','Fe']

glass_df = pd.DataFrame(data=X, columns=glass_identification.feature_names)
glass_df['target'] = y

X = glass_df[feature_cols]
y = glass_df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

st.title("Task 2 AI - Brent Peeters")

selected_option = st.selectbox("Selecteer een ML-algoritme", ["Decision Tree", "One VS One", "KNeigbors", "Alle algoritmes"])

regularisatieparameter = 1
AantalNeighbors = 1
criteria = 'entropy'

if(selected_option == "Decision Tree"):
    criteria = st.radio("Selecteer een criteria", ["entropy", "gini", "log_loss"])
elif(selected_option == "One VS One"):
    regularisatieparameter = int(st.text_input("Geef de beslissingsgrens aan (positief geheel getal): "))
elif(selected_option == "KNeigbors"):
    AantalNeighbors = int(st.text_input("Geef aan hoeveel Neighbors er mogen zijn (geheel getal): "))

else:
    criteria = st.radio("Selecteer een criteria", ["entropy", "gini", "log_loss"])
    regularisatieparameter = int(st.text_input("Geef de beslissingsgrens aan (positief geheel getal): "))
    AantalNeighbors = int(st.text_input("Geef aan hoeveel Neighbors er mogen zijn (geheel getal): "))
      
        

if st.button("Run Algoritme!"):

    # Display the selected option
    st.write("Geselecteerde optie:", selected_option)

    #3 ML techniques
    clf = DecisionTreeClassifier(criterion = criteria)
    decisionTree = clf.fit(X_train, y_train)
    decisionTree_pred = decisionTree.predict(X_test)

    clff = OneVsOneClassifier(LinearSVC(dual="auto", random_state=0, multi_class="ovr", C=regularisatieparameter))
    OneVsOne = clff.fit(X_train, y_train)
    OneVsOne_pred = OneVsOne.predict(X_test)

    nca = NeighborhoodComponentsAnalysis(random_state=42)
    knn = KNeighborsClassifier(n_neighbors=AantalNeighbors)
    nca_pipe = Pipeline([('nca', nca), ('knn', knn)])
    nca_pipe.fit(X_train, y_train)
    KNeighbors_pred = nca_pipe.predict(X_test)

    #confusion matrices
    conf_matrix_decisionTree = confusion_matrix(y_test, decisionTree_pred)
    conf_matrix_onevsone = confusion_matrix(y_test, OneVsOne_pred)
    conf_matrix_KNeighbors = confusion_matrix(y_test, KNeighbors_pred)

    #accurancy scores
    accuracy_score_decisionTree = metrics.accuracy_score(y_test, decisionTree_pred)
    accuracy_score_onevsone = metrics.accuracy_score(y_test, OneVsOne_pred)
    accuracy_score_KNeighbors = metrics.accuracy_score(y_test, KNeighbors_pred)

    if(selected_option == "Decision Tree"):
        st.text(conf_matrix_decisionTree)
        st.text(' ')
        st.text(f"Accurancy: {accuracy_score_decisionTree:<20.3f}")
    elif(selected_option == "One VS One"):
        st.text(conf_matrix_onevsone)
        st.text(' ')
        st.text(f"Accurancy: {accuracy_score_onevsone:<20.3f}")
    elif(selected_option == "KNeigbors"):
        st.text(conf_matrix_KNeighbors)
        st.text(' ')
        st.text(f"Accurancy: {accuracy_score_KNeighbors:<20.3f}")
    else:
        st.text("    " + "DecisionTree" + "              " + "OneVsOne" + "              " + "KNeighbors")
        st.text('')
        st.text(f"       {accuracy_score_decisionTree:<20.3f}    {accuracy_score_onevsone:<20.3f}   {accuracy_score_KNeighbors:<20.3f}")
        st.text('')
        for row1, row2, row3 in zip(conf_matrix_decisionTree, conf_matrix_onevsone , conf_matrix_KNeighbors):
            st.text("[" + " ".join([f"{value:2}" for value in row1]) + "]     [" + " ".join([f"{value:2}" for value in row2]) + "]     [" + " ".join([f"{value:2}" for value in row3]) + "]")