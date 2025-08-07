# brain_tumor_app.py
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input as mobilenet_preprocess
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess
import numpy as np
import matplotlib.pyplot as plt
import os
import tempfile
import base64
from fpdf import FPDF
from fpdf.enums import XPos, YPos
from PIL import Image
from collections import Counter

# Class names
class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']

# Medical information (replace with your actual medical content)
medical_info = {
    'glioma': {
        'title': "Glioma Tumor",
        'description': """Gliomas are brain tumors that arise from glial cells. They account for about 33% of all brain tumors.
        
**Characteristics:**
- Can occur in various brain regions
- Range from low-grade (slow-growing) to high-grade (fast-growing)
- May cause headaches, seizures, or neurological deficits

**Treatment Options:**
- Surgical removal
- Radiation therapy
- Chemotherapy
- Targeted drug therapy""",
        'color': '#FF6B6B'
    },
    'meningioma': {
        'title': "Meningioma Tumor",
        'description': """Meningiomas are tumors that arise from the meninges, the membranes surrounding the brain and spinal cord.
        
**Characteristics:**
- Usually benign (non-cancerous)
- Slow-growing
- More common in women
- May cause headaches, vision problems, or personality changes

**Treatment Options:**
- Observation for small tumors
- Surgical removal
- Radiation therapy""",
        'color': '#4ECDC4'
    },
    'notumor': {
        'title': "No Tumor Detected",
        'description': """The MRI scan shows no evidence of brain tumor.
        
**Recommendations:**
- Regular checkups if symptomatic
- Maintain healthy lifestyle
- Consult doctor if symptoms develop""",
        'color': '#45B7D1'
    },
    'pituitary': {
        'title': "Pituitary Tumor",
        'description': """Pituitary tumors are abnormal growths in the pituitary gland at the base of the brain.
        
**Characteristics:**
- May affect hormone production
- Can cause vision problems
- Often benign but can cause significant symptoms

**Treatment Options:**
- Medication to regulate hormones
- Surgical removal
- Radiation therapy""",
        'color': '#FFA07A'
    }
}

@st.cache_resource
def load_models():
    try:
        mobilenet_model = tf.keras.models.load_model("MobileNetV3_brain_tumor.keras")
        resnet_model = tf.keras.models.load_model("ResNet50_brain_tumor.keras")
        efficient_model = tf.keras.models.load_model("EfficientNetB0_brain_tumor.keras")
        return mobilenet_model, resnet_model, efficient_model
    except Exception as e:
        st.error(f"Failed to load models: {str(e)}")
        st.stop()

def preprocess_image(img_path, model_type):
    try:
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        x = np.expand_dims(img_array, axis=0)
        if model_type == 'mobilenet':
            return mobilenet_preprocess(x)
        elif model_type == 'resnet':
            return resnet_preprocess(x)
        elif model_type == 'efficientnet':
            return efficientnet_preprocess(x)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    except Exception as e:
        st.error(f"Image preprocessing failed: {str(e)}")
        st.stop()

def predict_with_all_models(img_path):
    mobilenet_model, resnet_model, efficient_model = load_models()
    x1 = preprocess_image(img_path, 'mobilenet')
    x2 = preprocess_image(img_path, 'resnet')
    x3 = preprocess_image(img_path, 'efficientnet')
    pred1 = mobilenet_model.predict(x1, verbose=0)[0]
    pred2 = resnet_model.predict(x2, verbose=0)[0]
    pred3 = efficient_model.predict(x3, verbose=0)[0]
    return pred1, pred2, pred3

def get_final_prediction(pred1, pred2, pred3):
    preds = [pred1, pred2, pred3]
    labels = [np.argmax(p) for p in preds]
    counter = Counter(labels)
    
    if len(counter) == 3:
        avg_confs = [np.mean([p[i] for p in preds]) for i in range(len(class_names))]
        final_idx = np.argmax(avg_confs)
        return final_idx, avg_confs[final_idx], preds
    else:
        final_idx = counter.most_common(1)[0][0]
        avg_conf = np.mean([p[final_idx] for p in preds])
        return final_idx, avg_conf, preds

def generate_pdf(pred_label, confidence, img_path, patient_info=None):
    try:
        pdf = FPDF()
        pdf.add_page()
        
        # Header
        pdf.set_font("Arial", 'B', 16)
        pdf.cell(200, 10, "Brain MRI Analysis Report", new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")
        pdf.ln(10)
        
        # Patient information
        if patient_info:
            pdf.set_font("Arial", size=12)
            pdf.cell(200, 10, f"Patient ID: {patient_info.get('id', 'N/A')}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.cell(200, 10, f"Age: {patient_info.get('age', 'N/A')}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.cell(200, 10, f"Sex: {patient_info.get('sex', 'N/A')}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.ln(10)
        
        # Results
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(200, 10, "Diagnosis Results", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, f"Prediction: {pred_label}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.cell(200, 10, f"Confidence: {confidence * 100:.2f}%", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.ln(10)
        
        # Image
        if os.path.exists(img_path):
            pdf.image(img_path, x=30, w=150)
            pdf.ln(10)
        
        # Medical information
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(200, 10, "Medical Information", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_font("Arial", size=12)
        pdf.multi_cell(0, 10, medical_info[pred_label]['description'])
        
        # Footer
        pdf.ln(10)
        pdf.set_font("Arial", 'I', 10)
        pdf.cell(0, 10, "This report was generated by AI and should be reviewed by a medical professional.", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        
        # Save to temporary file
        tmp_pdf_path = os.path.join(tempfile.gettempdir(), "brain_mri_report.pdf")
        pdf.output(tmp_pdf_path)
        
        # Create download link
        with open(tmp_pdf_path, "rb") as f:
            base64_pdf = base64.b64encode(f.read()).decode()
        
        os.unlink(tmp_pdf_path)
        href = f'<a href="data:application/octet-stream;base64,{base64_pdf}" download="brain_mri_report.pdf">📄 Download Full PDF Report</a>'
        return href
    except Exception as e:
        st.error(f"Failed to generate PDF: {str(e)}")
        return None

def main():
    st.set_page_config(layout="wide", page_title="Brain MRI Analysis", page_icon="🧠")
    st.sidebar.title("Navigation")
    
    if st.sidebar.button("🔄 Clear Cache"):
        st.cache_resource.clear()
        st.rerun()
    
    app_mode = st.sidebar.selectbox("Select Page", ["Home", "About", "MRI Analysis"])

    if app_mode == "Home":
        st.markdown("## **Brain Tumor MRI Classification**")
        st.markdown("Upload a brain MRI scan to get analysis results from multiple deep learning models.")
        st.markdown("""
        **Supported Tumor Types:**
        - Glioma
        - Meningioma
        - Pituitary Tumor
        - No Tumor (Normal Scan)
        """)

    elif app_mode == "About":
        st.markdown("### About This Application")
        st.markdown("""
        This app uses three state-of-the-art CNN models to classify brain MRI scans:
        - **MobileNetV3**: Fast and efficient model
        - **ResNet50**: Deeper architecture with higher accuracy
        - **EfficientNetB0**: Balanced approach for best results
        
        The system combines predictions from all models for more reliable diagnosis.
        """)
        st.markdown("""
        **Developed by ADITYA MALAV**
        """)

    elif app_mode == "MRI Analysis":
        st.header("Brain MRI Analysis")
        
        # Patient information
        with st.expander("Patient Information", expanded=True):
            col1, col2, col3 = st.columns(3)
            with col1:
                patient_id = st.text_input("Patient ID")
            with col2:
                patient_age = st.number_input("Age", min_value=1, max_value=120)
            with col3:
                patient_sex = st.selectbox("Sex", ["Male", "Female", "Other"])
        
        uploaded_file = st.file_uploader("Upload Brain MRI scan (JPEG/PNG):", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            try:
                file_ext = os.path.splitext(uploaded_file.name)[1].lower()
                with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    img_path = tmp_file.name
                
                st.image(img_path, caption="Uploaded MRI Scan", use_container_width=True)
                
                if st.button("🔍 Analyze MRI Scan", type="primary"):
                    with st.spinner("Processing with multiple AI models..."):
                        pred1, pred2, pred3 = predict_with_all_models(img_path)
                        final_idx, final_conf, all_preds = get_final_prediction(pred1, pred2, pred3)
                        final_label = class_names[final_idx]

                    # Display results
                    st.markdown("---")
                    st.subheader("Diagnosis Results")
                    
                    # Color-coded result box
                    color = medical_info[final_label]['color']
                    st.markdown(
                        f'<div style="background-color:{color};padding:20px;border-radius:10px;color:white;">'
                        f'<h2 style="color:white;text-align:center;">{medical_info[final_label]["title"]}</h2>'
                        f'<p style="text-align:center;font-size:24px;">Confidence: {final_conf*100:.2f}%</p></div>',
                        unsafe_allow_html=True
                    )

                    # Model predictions comparison
                    st.subheader("Model Predictions Comparison")
                    cols = st.columns(3)
                    model_names = ['MobileNetV3', 'ResNet50', 'EfficientNetB0']
                    for col, name, pred in zip(cols, model_names, [pred1, pred2, pred3]):
                        idx = np.argmax(pred)
                        col.metric(
                            label=name,
                            value=class_names[idx],
                            delta=f"{pred[idx]*100:.1f}%"
                        )

                    # Probability visualization
                    st.subheader("Detailed Probability Distribution")
                    fig, ax = plt.subplots(figsize=(10, 5))
                    width = 0.25
                    x = np.arange(len(class_names))
                    
                    for i, (name, pred) in enumerate(zip(model_names, [pred1, pred2, pred3])):
                        ax.bar(x + i*width, pred*100, width, label=name)
                    
                    ax.set_xticks(x + width)
                    ax.set_xticklabels(class_names)
                    ax.set_ylabel("Probability (%)")
                    ax.set_title("Model Confidence Across Tumor Types")
                    ax.legend()
                    st.pyplot(fig)

                    # Medical information
                    st.subheader(f"Clinical Information: {medical_info[final_label]['title']}")
                    st.markdown(medical_info[final_label]['description'])

                    # PDF Report
                    st.markdown("---")
                    st.subheader("Generate Medical Report")
                    patient_info = {
                        'id': patient_id,
                        'age': patient_age,
                        'sex': patient_sex
                    }
                    pdf_link = generate_pdf(final_label, final_conf, img_path, patient_info)
                    if pdf_link:
                        st.markdown(pdf_link, unsafe_allow_html=True)
                    
            except Exception as e:
                st.error(f"Error processing MRI scan: {str(e)}")
            finally:
                if 'img_path' in locals() and os.path.exists(img_path):
                    os.unlink(img_path)

if __name__ == "__main__":
    main()