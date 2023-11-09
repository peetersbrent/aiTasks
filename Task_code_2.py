from matplotlib import pyplot as plt
import streamlit as st
import pandas as pd

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

if st.button("Kies één van de verschillende ML algoritmes"):
    selected_option = st.radio(["Decision Tree", "One VS One", "KNeigbors", "Alle algoritmes"])

    # Display the selected option
    st.write("You selected:", selected_option)

    #3 ML techniques
    clf = DecisionTreeClassifier(criterion = "entropy")
    decisionTree = clf.fit(X_train, y_train)
    decisionTree_pred = decisionTree.predict(X_test)

    clff = OneVsOneClassifier(LinearSVC(dual="auto", random_state=0, multi_class="ovr"))
    OneVsOne = clff.fit(X_train, y_train)
    OneVsOne_pred = OneVsOne.predict(X_test)

    nca = NeighborhoodComponentsAnalysis(random_state=42)
    knn = KNeighborsClassifier(n_neighbors=1)
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
        st.text("Accurancy:", accuracy_score_decisionTree)
    elif(selected_option == "One VS One"):
        st.text(conf_matrix_onevsone)
        st.text(' ')
        st.text("Accurancy:", accuracy_score_onevsone)
    elif(selected_option == "KNeigbors"):
        st.text(conf_matrix_KNeighbors)
        st.text(' ')
        st.text("Accurancy:", accuracy_score_KNeighbors)
    else:
        st.text("    " + "DecisionTree" + "              " + "OneVsOne" + "              " + "KNeighbors")
        st.text('')
        st.text(f"       {accuracy_score_decisionTree:<20.3f}    {accuracy_score_onevsone:<20.3f}   {accuracy_score_KNeighbors:<20.3f}")
        st.text('')
        for row1, row2, row3 in zip(conf_matrix_decisionTree, conf_matrix_onevsone , conf_matrix_KNeighbors):
            st.text("[" + " ".join([f"{value:2}" for value in row1]) + "]     [" + " ".join([f"{value:2}" for value in row2]) + "]     [" + " ".join([f"{value:2}" for value in row3]) + "]")