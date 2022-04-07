import streamlit as st 

st.title('Model in application')

st.write("Hello welcome !")

@st.cache
def load_model():
    import joblib

    model = joblib.load("linear_regression.joblib")
    return model

model = load_model()

X1 = st.number_input("enter X1")
X2 = st.number_input("enter X2")

prediction = model.predict([[X1, X2]])
st.write("the prediction is: {}".format(prediction))


