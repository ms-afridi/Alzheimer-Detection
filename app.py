import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
from streamlit_option_menu import option_menu
import base64
from fpdf import FPDF
import re

import os
import gdown
import tensorflow as tf

model_path = 'my_model.h5'
model_url = 'https://drive.google.com/uc?id=1Yy80wy8w6P3Ab8Yy1vKnbq0wz8cX3o6u'

if not os.path.exists(model_path):
    print("Downloading model...")
    gdown.download(model_url, model_path, quiet=False)

model = tf.keras.models.load_model(model_path)

# Class labels
class_labels = ['Mild Demented', 'Moderate Demented', 'Non Demented', 'Very Mild Demented']

# Dummy doctor database
doctor_data = {
    'Delhi': [
        'Dr. Anil Kumar - AIIMS, Delhi - 011-26588500',
        'Dr. Ritu Sharma - Fortis, Delhi - 011-47132222'
    ],
    'Mumbai': [
        'Dr. Sneha Verma - Kokilaben Hospital, Mumbai - 022-30999999',
        'Dr. Rajiv Mehta - Lilavati Hospital, Mumbai - 022-26751000'
    ],
    'Bangalore': [
        'Dr. Prakash Rao - NIMHANS, Bangalore - 080-26995001',
        'Dr. Anita Menon - Apollo Hospitals, Bangalore - 080-40304030'
    ]
}

# Preprocess uploaded image
def preprocess_image(image):
    image = image.convert('RGB')
    image = image.resize((176, 176))
    image = np.array(image) / 255.0
    return np.expand_dims(image, axis=0)

# Validation
def validate_phone_number(phone_number):
    return bool(re.match(r'^\d{10}$', str(phone_number)))

def validate_name(name):
    return all(char.isalpha() or char.isspace() for char in name)

def validate_input(name, age, contact, file):
    return all([name, age, contact, file])

# Generate PDF report
def create_pdf(name, age, gender, contact, city, prediction_label, image, recommend_doctors):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font('Times', 'B', 24)
    pdf.cell(200, 20, 'Alzheimer Detection Report', 0, 1, 'C')

    pdf.set_font('Arial', 'B', 16)
    pdf.cell(200, 10, 'Patient Details', 0, 1)
    pdf.set_font('Arial', '', 12)
    pdf.cell(200, 10, f'Name: {name}', 0, 1)
    pdf.cell(200, 10, f'Age: {age}', 0, 1)
    pdf.cell(200, 10, f'Gender: {gender}', 0, 1)
    pdf.cell(200, 10, f'Contact: {contact}', 0, 1)
    pdf.cell(200, 10, f'City: {city}', 0, 1)

    # Save and add MRI scan image
    png_file = "uploaded_image.png"
    image.save(png_file, "PNG")
    pdf.cell(200, 10, 'MRI Scan:', 0, 1)
    pdf.image(png_file, x=40, y=100, w=50, h=50)
    pdf.ln(60)

    pdf.set_font('Arial', 'B', 16)
    pdf.cell(200, 10, f'Prediction: {prediction_label}', 0, 1)

    if prediction_label != "Non Demented":
        pdf.set_text_color(255, 0, 0)
        pdf.cell(200, 10, 'Dementia detected. Please consult a neurologist.', 0, 1)
        pdf.set_text_color(0, 0, 255)
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(200, 10, 'Recommended Doctors:', 0, 1)
        for doc in recommend_doctors:
            pdf.set_font('Arial', '', 12)
            pdf.cell(200, 10, doc, 0, 1)
    else:
        pdf.set_text_color(0, 128, 0)
        pdf.cell(200, 10, 'No signs of dementia detected.', 0, 1)

    return pdf.output(dest="S").encode("latin-1")

# PDF download helper
def create_download_link(val, filename):
    b64 = base64.b64encode(val)
    return f'<a href="data:application/octet-stream;base64,{b64.decode()}" download="{filename}.pdf">Download Report</a>'

# Sidebar navigation
selected = option_menu(
    menu_title=None,
    options=["Home", "Alzheimer Detection", "About Us"],
    icons=["house", "search", "info-circle"],
    orientation="horizontal",
)

# Home page
if selected == "Home":
    st.title("Alzheimer's Detection System")
    st.write("""
    Alzheimer's disease is a progressive disorder that causes brain cells to degenerate and die. 
    This app uses MRI images to predict the stage of Alzheimer's disease using a trained deep learning model.
    """)

# About Us
elif selected == "About Us":
    st.title("About the Project")
    st.write("This mini project was developed by Md Shahid Afridi. The application uses a CNN model to classify MRI scans into stages of Alzheimer's disease.")
    st.write("For educational and demonstration purposes only.")

# Alzheimer Detection
elif selected == "Alzheimer Detection":
    st.title("Alzheimer Detection")
    st.write("Enter your details and upload an MRI scan to get a prediction.")

    with st.form("input_form"):
        name = st.text_input("Name")
        age = st.number_input("Age", 1, 120, 40)
        gender = st.radio("Gender", ("Male", "Female", "Other"))
        contact = st.text_input("Contact Number")
        city = st.selectbox("Select your city", list(doctor_data.keys()))
        file = st.file_uploader("Upload MRI Image", type=['jpg', 'jpeg', 'png'])
        submit = st.form_submit_button("Submit")

    if submit:
        if not validate_input(name, age, contact, file):
            st.error("Please fill in all fields and upload an image.")
        elif not validate_name(name):
            st.error("Invalid name. Please avoid numbers or special characters.")
        elif not validate_phone_number(contact):
            st.error("Invalid contact number. Must be 10 digits.")
        else:
            st.success("Details submitted successfully!")
            image = Image.open(file)
            st.image(image, caption="Uploaded MRI", width=200)

            processed = preprocess_image(image)
            prediction = np.argmax(model.predict(processed), axis=1)[0]
            prediction_label = class_labels[prediction]

            st.success(f"Predicted Class: {prediction_label}")

            recommend_doctors = doctor_data.get(city, [])
            if prediction_label != "Non Demented":
                st.warning("Dementia detected. Consult a neurologist.")
                st.subheader(f"Doctors in {city}:")
                for doc in recommend_doctors:
                    st.write(f"👨‍⚕️ {doc}")
            else:
                st.success("No signs of dementia detected.")

            if st.button("Generate PDF Report"):
                pdf_content = create_pdf(name, age, gender, contact, city, prediction_label, image, recommend_doctors)
                st.markdown(create_download_link(pdf_content, "Alzheimer_Report"), unsafe_allow_html=True)