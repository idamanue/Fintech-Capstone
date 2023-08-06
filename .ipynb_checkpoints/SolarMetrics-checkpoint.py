# Importing the Project libraries
#import solar_metrics_lib as sml

import pandas as pd
import numpy as np

import streamlit as st
#from streamlit_option_menu import option_menu
from PIL import Image
import plotly.express as px

# import our data as a pandas dataframe
mlr_df = pd.read_csv("metrics/linear_regression.csv", usecols=range(1,5))
dnn_df = pd.read_csv("metrics/deep_learning.csv", usecols=range(1,5))

# Building a dataframe of panels
# Future improvement: data will be taken from the user and saved in a database
def panel_dataframe():
    solar_panel= {
        'module_id':["PHL6621","CRS0018","QCL0182","QCL0201","QCL0346"],
        'stc_power' :[369.6,382.6,382.78,376.66,446.39],
        'gamma':[-0.3935,-0.3779,-0.3730, -0.3637,-0.3642],
        'area':['1.98','2.06','1.895', 1.895,2.239],
        'cec_noct':[47.8,47.5,46.3,46.3,46.5]
        }
    df = pd.DataFrame(solar_panel)
    
    return df

# set up the app with wide view preset and a title
st.set_page_config(layout="wide")
st.title("Which Solar Panels to Choose for my Solar System?")

# import our data as a pandas dataframe
df = panel_dataframe()

# get a list of all available panels and ref noct, for the widgets
module_list = list(df['module_id'].unique())
cec_list = list(df['cec_noct'].unique())

# put all widgets in sidebar and have a subtitle
with st.sidebar:
    st.subheader("Select Panels to Compare")
    # widget to choose which solar panels to compare. You can select at most 3
    solar_panels = st.multiselect(label = "Choose a solar panel", options = module_list, max_selections=3)
    # widget to choose which Estimator to use
    estimator = st.selectbox(label = "Choose an estimator", options = ['Deep Learning','Multiple Linear Regression'])
    # widget to choose which metric to display: PTC, NOCT, ROI
    #solar_panel_2 = st.selectbox(label = "Choose a solar panel", options = metric_list)


if estimator== 'Deep Learning':
    estimator_df = dnn_df.copy()
else:
    estimator_df = mlr_df.copy()


#combined_data = sml.get_csv(solar_panel_1)
#combined_data = sml.get_csv(solar_panel_1)
#combined_df = sml.preprocessing(combined_data)
#mlr = sml.multiple_linear_regression(combined_df, solar_panel_1)
    
# use selected values from widgets to filter dataset down to only the rows we need
#query = f"module_id=='{solar_panels[0]}' | module_id=='{solar_panels[1]}' | module_id=='{solar_panels[2]}'"
#df_filtered = df.query(query)
df_filtered = estimator_df.query("module_id == @solar_panels")
st.write(df_filtered)
#st.write(combined_df)
#st.write(mlr)

'''
with st.sidebar:
    selected = option_menu(
        menu_title = "Solar Metrics",
        options = [ "Home", "Estimator", "Contacts"],
        icons = ['house', 'calculator', 'envelope'],
        menu_icon = "cast",
        default_index = 0
    )
    
if selected == "Home":
    st.title(" Solar Metrics")
    st.subheader("Unlock the Power of Savings with NOCT: Harnessing Accurate Calculations for Solar Modules!")
    st.write("When it comes to saving money through solar energy, understanding and utilizing the concept of Nominal Operating Cell Temperature (NOCT) is crucial. NOCT refers to the temperature at which a solar module operates under real-world conditions, taking into account factors such as ambient temperature, solar radiation, wind speed, and module design.")
    
    image = Image.open('image1.png')
    st.image(image, caption='Your payback time reduces by half! sources: Solar-Estimate.org')
    
    


    

    
if selected == "Estimator":
    st.header("Estimate your NOCT value to determine your solar panel performance and efficiency in real-world conditions.")
    st.subheader("Please enter values below and find your Power Conversion Technology ")
    st.write("PTC refers to the efficiency and effectiveness of converting sunlight into usable electrical energy by the solar panel system. These information are located on your solar pannel's label")
    input_value = st.text_input("Module Technology")
    input_value = st.text_input("Model Number")
    input_value = st.text_input("maximum power at Standard Test Conditions")
    input_value = st.text_input("Temperature Coefficient of Pmax")
    input_value = st.text_input("Total Module Area")
    input_value = st.text_input("Contact Passivated Tunneling (CPT)")
    input_value = st.text_input("CEC NOCT vs Solar Metrics NOCT")
    


    st.subheader("Please enter the values below to determine Return On Investment")
    st.write("Estimation in years")
    input_value = st.text_input("System Size (Watt)")
    input_value = st.text_input("Tax Credit")
    input_value = st.text_input("Average Monthly electricity bill")


    
if selected == "Contacts":
    st.title(" Project Participants and Contacts")
    st.subheader("FINTECH Capstone Project - ASU Bootcamp")
    st.write(f"Joseph Kuitche: Kuitche@asu.edu")
    st.write(f"Manuela Nkwinkwa: mnkwinkwa@gmail.com")
    st.subheader(f"Sources")
    st.write('www.solar.com')
    st.write('www.solar-estimate.org')
    st.write('www.iask.ai')
    st.write('https://www.youtube.com/watch?v=01SeFqQOjBM')


'''
