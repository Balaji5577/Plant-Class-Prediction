import numpy as np
import pandas as pd
import streamlit as st
from datetime import date
import pickle
from joblib import load
from streamlit_option_menu import option_menu

# st.set_page_config(layout="wide")
st.set_page_config(page_title="Zeal CRE",layout="wide",page_icon="🌱")
st.title(":black[Plant Class Prediction]")

def Class():
# Define dictionaries for mapping strings to integers
    Random_values_dict = {'R1': 0, 'R2': 1, 'R3': 2}

    Random_values= ['R1','R2','R3']

    with st.form('Regression'):
        col1,col2 = st.columns([0.5,0.5])

        with col1:
                    Random = st.selectbox(label='Random', options=Random_values)
                    ACHP = st.number_input(label='Average of Chlorophyll in the Plant (ACHP)' ,min_value= 32.66, max_value= 46.43)
                    PHR = st.number_input(label='Plant Height Rate (PHR)', min_value=37.02, max_value=77.04)
                    AWWGV = st.number_input(label='Average Wet Weight of the Growth Vegetative (AWWGV)', min_value=0.84, max_value=1.77)
                    ALAP = st.number_input(label='Average Leaf Area of the Plant (ALAP)', min_value=658.48, max_value=1751.03)
                    ANPL = st.number_input(label='Average Number of Plant Leaves (ANPL)', min_value=2.95, max_value=5.03)
                    ARD = st.number_input(label='Average Root Diameter (ARD)', min_value=11.07, max_value=23.32)
                    
        with col2:
                    AADWR = st.number_input(label='Average Dry Weight of the Root (AADWR)', min_value=0.24, max_value=2.19)
                    PDMVG = st.number_input(label='Percentage of Dry Matter for Vegetative Growth (PDMVG)', min_value=8.02, max_value=43.66)
                    ARL = st.number_input(label='Average Root Length (ARL)', min_value=12.35, max_value=23.25)
                    AWWR = st.number_input(label='Average Wet Weight of the Root (AWWR)', min_value=1.12, max_value=4.77)
                    ADWV = st.number_input(label='Average Dry Weight of Vegetative Plants (ADWV)', min_value=0.03, max_value=0.68)
                    PDMRG = st.number_input(label='Percentage of Dry Matter for Root Growth (PDMRG)', min_value=23.63, max_value=46.57)
                    
                    button = st.form_submit_button(label='SUBMIT')

        if button: 
                
                with open(r"D:/Project/Zeal/Class.pkl", 'rb') as f:     # Adjust the file path as per your saved model
                        status_model = pickle.load(f)

                # Convert status and item_type to their corresponding integer representations
                Random_values_int = Random_values_dict.get(Random)

                y_pred = status_model.predict(np.array([[Random_values_int, ACHP, PHR, AWWGV, ALAP, ANPL, ARD, AADWR, PDMVG,ARL,AWWR,ADWV,PDMRG]]))
            
                status = y_pred[0]

                if status == 0:
                    st.header('Predicted Plant Class is "SA"')
                
                elif status == 1:
                    st.header('Predicted Plant Class is "SB"')

                elif status == 2:
                    st.header('Predicted Plant Class is "SC"')

                elif status == 3:
                    st.header('Predicted Plant Class is "TA"')

                elif status == 4:
                    st.header('Predicted Plant Class is "TB"')

                else:
                    st.header('Predicted Plant Class is "TC"')
                    
                

def home():

     
    st.header(' :green[Importance of Machine Learning in Plant Class Prediction]')
    st.write('### :violet[Introduction:]')
    st.write('Welcome to the Plant Class Web Page! This platform utilizes machine learning algorithms to provide valuable insights into details of plants and Class of Plant.') 
    
    st.write('### :violet[Importance of Machine Learning:]')
    st.write('In the Agricultural field, accurately predicting the class of Plant is paramount.')
    st.write('Now with advanced technologies like machine learning helps by making the job little easier. By leveraging vast amounts of historical datas and') 
    st.write('sophisticated algorithms, machine learning can uncover hidden patterns and make highly accurate predictions.')
    
        
    st.header(' :green[Use Case of This Project:]')
    st.write('Our project focuses on the aspect of:')

    st.write('### :violet[Plant Class Prediction:]')
    st.write('Users can give input various parameters related to Plants measurement and our machine learning model predicts the Class of the Plant which it belongs to.')  

option = option_menu("", ['Home','Plant Class Prediction'],default_index=0,orientation="horizontal")

if option == 'Home':
    home()

elif option == 'Plant Class Prediction':
    Class()

