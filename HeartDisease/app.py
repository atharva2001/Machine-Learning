import pickle
import streamlit as st

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)


st.title('Heart Disease Prediction App Using Logistic Regression')

st.write('Enter the details of the patient to predict the probability of heart disease.')

age = st.number_input('Enter age', 0, 100)
gender = st.radio("Choose the gender", 
                  [1, 0], 
                  captions=["Male", "Female"])
cp = st.radio("Choose the chest pain type", 
              [0, 1, 2, 3],
              captions=["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"])
trestbps = st.number_input('Enter resting blood pressure', placeholder=145)
chol = st.number_input('Enter cholesterol', placeholder=174)
fbs = st.radio("Choose the fasting blood sugar", 
               [1, 0], 
               captions=["True", "False"])
restecg = st.radio("Choose the resting electrocardiographic results", 
                   [0, 1, 2],
                   captions=["Normal", "Having ST-T wave abnormality", "Showing probable or definite left ventricular hypertrophy"])
thalach = st.number_input('Enter maximum heart rate achieved', placeholder=125)
exang = st.radio("Choose exercise induced angina", 
                 [1, 0], 
                 captions=["Yes", "No"])
oldpeak = st.number_input('Enter ST depression induced by exercise relative to rest', placeholder=2.6)
slope = st.radio("Choose the slope of the peak exercise ST segment", 
                 [0, 1, 2],
                 captions=["Upsloping", "Flat", "Downsloping"])
ca = st.radio("Choose the number of major vessels colored by flourosopy", 
              [0, 1, 2, 3],
              captions=["0", "1", "2", "3"])
thal = st.radio("Choose the thalassemia", 
                [1, 2, 3],
                captions=["Normal", "Fixed Defect", "Reversable Defect"])


res = str(model.predict([[age, gender, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])[0])

if st.button("Predict Heart Disease", type="primary"):
    st.write("Heart Disease Prediction:", res)
