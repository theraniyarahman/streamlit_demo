import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import streamlit as st
import pickle




st.title("Decision Tree Classifier using Titanic Dataset")

with st.sidebar:
     options = st.radio("Options",["Model","Analytics"])

if options =="Model":

    Pclass   =  st.number_input("Enter value for Pclass: ", min_value=None, max_value=None,
                               value=None, step=None, format=None, key=None, help=None,
                               on_change=None, args=None, kwargs=None,placeholder=None,
                               disabled=False, label_visibility="visible", icon=None, 
                               width="stretch", bind=None, persist_state=None)


    Age      = st.number_input("Enter value for Age: ", min_value=None, max_value=None,
                               value=None, step=None, format=None, key=None, help=None,
                               on_change=None, args=None, kwargs=None,placeholder=None,
                               disabled=False, label_visibility="visible",icon=None, 
                               width="stretch", bind=None, persist_state=None)
    

    Fare     = st.number_input("Enter value for Fare: ", min_value=None, max_value=None,
                               value=None, step=None, format=None, key=None, help=None,
                               on_change=None, args=None, kwargs=None,placeholder=None,
                               disabled=False, label_visibility="visible", icon=None, 
                               width="stretch", bind=None, persist_state=None)


    Alone      =  st.radio('Sex', ['Yes', 'No'], index=0, key=None, help=None, on_change=None, 
                                     args=None, kwargs=None, disabled=False, horizontal=False,
                                     captions=None, label_visibility="visible",
                                     width="content", bind=None, persist_state=None)


    sex      =  st.radio('Sex', ['Male', 'Female'], index=0, key=None, help=None, on_change=None, 
                             args=None, kwargs=None, disabled=False, horizontal=False,
                             captions=None, label_visibility="visible",
                             width="content", bind=None, persist_state=None)

    if sex == 'Male':
        Sex_male = True
        Sex_female = False
    elif sex == 'Female':
        Sex_male = False
        Sex_female = True

    if Alone == 'Yes':
        alone = True
    else:
        alone = False

    features = np.array([Pclass, Age, Fare, alone, Sex_female, Sex_male], ndmin = 2)
    features_df = pd.DataFrame(features, columns = ['Pclass', 'Age', 'Fare', 'Alone', 'Sex_female', 'Sex_male'])

    with open('model.pkl','rb') as  f:
        model = pickle.load(f)

    if st.button('predict'):
        predicted = model.predict(features_df)

        if predicted[0] == 1:
            st.markdown('Survived')
        else:
            st.markdown('Not Survived')

