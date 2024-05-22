import pandas as pd
import streamlit as st
import pickle

# Load your pre-trained Alzheimer's model
model = pickle.load(open('gradexp.pkl', 'rb'))

def main(): 
    st.title("Alzheimer's Disease Detection")

    def home_page():
    st.image(BANNER)
    st.write("""
        ## Introduction to Alzheimer's Disease
        Alzheimer's disease (AD) is a progressive neurodegenerative disease. Though best known for its role in declining memory function, symptoms also include: difficulty thinking and reasoning, making judgements and decisions, and planning and performing familiar tasks. It may also cause alterations in personality and behavior. The cause of AD is not well understood. There is thought to be a significant hereditary component. For example, a variation of the APOE gene, APOE e4, increases risk of Alzheimer's disease.

        ## Why Early Detection Matters
        Early detection of Alzheimer's disease is paramount because it offers the best chance for effective treatment and improved quality of life. Identifying the condition at its onset allows for timely interventions, which can slow its progression and enable individuals and their families to plan for the future. Early detection also facilitates access to support services and clinical trials, fostering hope for more effective therapies in the fight against this devastating disease.

        ## Purpose of the project
        The purpose of this project proposal is to develop a machine learning model for the early prediction of Alzheimer's disease. Alzheimer's disease is a devastating neurodegenerative disorder that affects millions of individuals worldwide. Early detection is crucial for better patient care and the development of potential interventions. This project aims to leverage machine learning techniques to create a predictive model that can identify individuals at risk of Alzheimer's disease based on relevant data.
        
        <br>
                
        """, unsafe_allow_html=True)

    st.caption('Finished reading? Navigate to the `Prediction Page` to make some predictions')
    
    html_temp = """
    <div style="background:#025246 ;padding:10px">
    <h2 style="color:white;text-align:center;">Alzheimer's Disease Prediction App</h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)

    # Create input fields based on your Alzheimer's dataset
    st.sidebar.subheader("User Input Features")
    age = st.sidebar.slider("Age", min_value=60, max_value=90, value=60)
    pt_educat = st.sidebar.slider("Years of Education", min_value=0, max_value=23, value=0)
    cdrsb = st.sidebar.selectbox("Clinical Dementia Rating Sum of Boxes", options=[0.0, 0.5, 1.0, 2.0], index=0)
    adas13 = st.sidebar.number_input("Alzheimer's Disease Assessment Scale - Cognitive Subscale 13", min_value=0.0, max_value=50.0, value=0.0)
    adas11 = st.sidebar.number_input("Alzheimer's Disease Assessment Scale - Cognitive Subscale 11", min_value=0.0, max_value=50.0, value=0.0)
    faq = st.sidebar.number_input("Functional Activities Questionnaire", min_value=0.0, max_value=10.0, value=0.0)
    ldeltotal = st.sidebar.number_input("Total Learning and Delayed Recall Score", min_value=0.0, max_value=50.0, value=0.0)
    mmse = st.sidebar.slider("Mini-Mental State Examination", min_value=0, max_value=30, value=0)
    adasq4 = st.sidebar.number_input("Alzheimer's Disease Assessment Scale - Question 4", min_value=0.0, max_value=50.0, value=0.0)
    ravlt_immediate = st.sidebar.number_input("Rey Auditory Verbal Learning Test - Immediate Recall", min_value=0.0, max_value=100.0, value=0.0)

    
    # Prediction button
    if st.button("Predict"):
        try:
            # Organize the inputs
            input_data = pd.DataFrame([[age, pt_educat, cdrsb, adas13, adas11, faq, ldeltotal, mmse, adasq4, ravlt_immediate]], 
                                      columns=['AGE', 
                                               'PTEDUCAT', 
                                               'CDRSB', 
                                               'ADAS13', 
                                               'ADAS11',
                                               'FAQ', 
                                               'LDELTOTAL', 
                                               'MMSE', 
                                               'ADASQ4', 
                                               'RAVLT_immediate'])

            # Make prediction
            prediction = model.predict(input_data)
            output = prediction[0]

            # Display the prediction
            if output == 0:
                st.success('The predicted status is: Healthy (No sign of Alzheimer’s Disease)', icon="✅")
                st.markdown("<div style='background-color: #d4edda; padding: 10px; border-radius: 5px;'><h4 style='color: #155724;'>Healthy (No sign of Alzheimer’s Disease)</h4></div>", unsafe_allow_html=True)
            elif output == 1:
                st.warning('The predicted status is: Mild Cognitive Impairment', icon="⚠️")
                st.markdown("<div style='background-color: #fff3cd; padding: 10px; border-radius: 5px;'><h4 style='color: #856404;'>Mild Cognitive Impairment</h4></div>", unsafe_allow_html=True)
            elif output == 2:
                st.error('The predicted status is: Demented (Alzheimer’s Disease)', icon="❌")
                st.markdown("<div style='background-color: #f8d7da; padding: 10px; border-radius: 5px;'><h4 style='color: #721c24;'>Demented (Alzheimer’s Disease)</h4></div>", unsafe_allow_html=True)
        except Exception as e:
            st.error(f"An error occurred: {e}")

if __name__ == '__main__': 
    main()
