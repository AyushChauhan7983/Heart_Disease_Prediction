import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import streamlit as st

heart_data = pd.read_csv("heart.csv")
st.title("Heart disease prediction")

Age = st.number_input('Age')
Sex = st.number_input('Sex (1-Male, 0-Female)')
Cp = st.number_input('Chest Pain Type(0,1,2,3)')
trestbps = st.number_input('Resting Blood Pressure (in mm Hg on admission to the hospital)')
chol = st.number_input('Serum Cholestral in mg/dl')
fbs = st.number_input('Fasting Blood Sugar (1->True, 0->False)')
restecg = st.number_input('Resting Electrocardiographic Results')
thalach = st.number_input('Maximum Heart Rate achieved')
exang = st.number_input('Exercise Induced Angina (1 = yes; 0 = no)')
oldpeak = st.number_input('ST depression induced by exercise relative to rest')
slope = st.number_input('The Slope of the peak exercise ST segment')
ca = st.number_input('Number of major vessels (0-3) colored by flourosopy')
thal = st.number_input('1 = Normal, 2 = Fixed defect, 3 = Reversable defect')

X = heart_data.drop(columns='target',axis=1)
Y = heart_data['target']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

#Logistic Regression
model = LogisticRegression()
model.fit(X_train, Y_train)
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)

if st.button("Want to test"):
    input_data = ([Age, Sex, Cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal])
    input_data_as_numpy_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    prediction = model.predict(input_data_reshaped)
    if (prediction[0] == 0):
        st.success("The person does not have a heart disease")
    else:
        st.error("The person has heart disease")

