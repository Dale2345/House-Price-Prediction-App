# Importing requisite libraries
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Page configuration
st.set_page_config(
     page_title='House Price Prediction App',
     page_icon='🏠',
     layout='wide',
     initial_sidebar_state='expanded')

# Title of the app
st.title('🏠 House Price Prediction App')

# Load dataset
df = pd.read_csv('csvdata.csv')  

# Input widgets
st.sidebar.subheader('Input Features')
area = st.sidebar.slider('Area (sqft)', int(df.Area.min()), int(df.Area.max()), int(df.Area.mean()))
no_of_bedrooms = st.sidebar.slider('No. of Bedrooms', 1, 10, 2)

# Prepare data for modeling
X = df[['Area', 'No. of Bedrooms']]
y = df['Price']

# Data splitting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model building
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Apply model to make predictions
predicted_price = model.predict([[area, no_of_bedrooms]])

# Print input features
st.subheader('Input Features')
input_feature = pd.DataFrame([[area, no_of_bedrooms]],
                             columns=['Area (sqft)', 'No. of Bedrooms'])
st.write(input_feature)

# Print prediction output
st.subheader('Predicted Price')
st.metric('Price', f'₹{predicted_price[0]:,.2f}', '')

