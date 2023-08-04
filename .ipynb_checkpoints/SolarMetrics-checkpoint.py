import streamlit as st
from streamlit_option_menu import option_menu
from PIL import Image


    
#adding a side bar

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



