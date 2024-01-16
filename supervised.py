import streamlit as st
import pandas as pd
import pycaret.classification as c
import pycaret.regression as r
def train(data, target,_type):
    
    a = r if _type=="Regression" else c
    try:  
        a.setup(data=data,target=target)  
        modelsList = a.models().Name.values
        chosenModel = st.selectbox("Choose a model:", modelsList)
        if st.button("Train Model"):    
            myModel = a.create_model(a.models()[a.models().Name == chosenModel].index[0]) 
            a.predict_model(myModel,data.drop(columns=[target]))
            st.dataframe(a.pull().loc["Mean"])
    except ValueError:
        st.write(f"This variable throws an error is not suitable to be a target variable for {_type}")
        st.write(ValueError)


st.title("Supervised Project 📊📈")
st.subheader("Choose a CSV file 😊")
uploaded_file = st.file_uploader("Choose a file",type="csv")
if uploaded_file is not None:

    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
        st.dataframe(df.head())
        st.write("Whats your target variable?")
        # df.columns
        problemOptions=["Classification","Regression"]
        choices = df.columns[::-1]
        target = st.selectbox("Choose a Target Variable:", choices)
        problemType ="Regression" if df[target].dtype in ['float16', 'float32', 'float64'] else "Classification"
        
        st.write("Target Variable type:" ,df[target].dtype )
        # st.text("Problem type:" ,problemType)
        st.write("Problem type:" ,problemType)
        problemType=st.radio(f"recommended problemType {problemType}:", problemOptions ,index=problemOptions.index(problemType))
        
        # assigning model
        # model(df,target,problemType)
        
        train(df,target,problemType)
        
        
        # st.tk.Button(root, text="Apply Model")
        
    else:
        st.write("Unsupported file type.")



