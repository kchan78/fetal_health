# App to predict the class of fetal health
# Using a pre-trained ML model in Streamlit

# Import libraries
import streamlit as st
import pandas as pd
import pickle
import sklearn

st.title('Fetal Health Classification: A Machine Learning App') 

# Display the image
st.image('fetal_health_image.gif', width = 650)

st.subheader("This machine learning application uses multiple inputs to predict fetal health classifications. ")
st.write("Upload your dataset to get started! Please ensure your data strictly adheres to the specified format, shown below:") 
# Load and display original dataset
original_df = pd.read_csv('fetal_health.csv')
st.dataframe(original_df.head())

fetal_health_file = st.file_uploader('Upload your own fetal health data')

# Reading the pickle files that we created before 
# Random Forest
rf_pickle = open('rf_fetal.pickle', 'rb') 
# Map file
map_pickle = open('output_fetal.pickle', 'rb') 
unique_fetal_mapping = pickle.load(map_pickle) 
rf_model = pickle.load(rf_pickle)
map_pickle.close()
rf_pickle.close()

# color function definition 
# NOTE: Code from https://www.geeksforgeeks.org/highlight-pandas-dataframes-specific-columns-using-applymap/
def highlight_cols(s): 
    if s == 'Normal':
        color = 'lime'
    elif s == 'Suspect':
        color = 'yellow'
    else:
        color = 'orange'
    return 'background-color: % s' % color 

if fetal_health_file is None:
    pass
else: 
    # Loading user data
    user_df = pd.read_csv(fetal_health_file) # User provided data

    # Dropping null values
    user_df = user_df.dropna() 

    # Predictions for user data
    user_pred = rf_model.predict(user_df).astype(int)

    # Predicted health
    user_pred_health = unique_fetal_mapping[user_pred-1]    # NOTE: subtrack 1 because it predicts as 1,2,3, not 0,1,2

    # Prediction Probabilities
    user_pred_prob = rf_model.predict_proba(user_df)

    # Adding predicted health to user dataframe
    user_df['Predicted Fetal Health'] = user_pred_health
    # Storing the maximum prob. (prob. of predicted health) in a new column
    user_df['Prediction Probability'] = user_pred_prob.max(axis = 1)

    # Apply color to the prediction column
    user_df = user_df.style.applymap(highlight_cols,
                                     subset = pd.IndexSlice[:, ['Predicted Fetal Health']])
    # Show the predicted health on the app
    st.subheader("Predicting Fetal Health")
    st.dataframe(user_df)

# Showing additional items
st.subheader("Prediction Performance")
tab1, tab2, tab3 = st.tabs(["Feature Importance", "Confusion Matrix", "Classification Report"])

with tab1:
  st.image('feature_imp.svg')
with tab2:
  st.image('class_matrix.svg')
with tab3:
    df = pd.read_csv('rf_class_report.csv', index_col=0)
    st.dataframe(df)
