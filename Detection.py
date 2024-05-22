import pandas as pd
import streamlit as st
import pickle

# Load your pre-trained Alzheimer's model
model = pickle.load(open('gradexp.pkl', 'rb'))

def main(): 
    st.title("Alzheimer's Disease Detection")

    st.write("""
        ## Alzheimer's Disease: A Neurodegenerative Challenge
        Alzheimer's disease (AD) is a devastating neurodegenerative condition characterised by progressive cognitive decline and memory loss. While memory impairment is a hallmark symptom, AD also affects reasoning, decision-making, and the ability to perform daily tasks. This multifaceted disease can even alter personality and behavior. Though the underlying causes are not fully understood, genetic factors like the APOE e4 gene variant play a significant role in increasing AD risk. With an aging global population, enhancing early detection and diagnosis of AD is crucial for timely intervention and improving patient outcomes through this research on ensemble machine learning models.

        ## Why Early Detection Matters
        Achieving early detection of AD is pivotal as it unlocks the best opportunities for effective treatment and improved quality of life for patients. Identifying AD at its onset enables timely clinical interventions that can significantly slow the progression of this debilitating condition. Early diagnosis provides a critical window for individuals and families to plan and access support services before symptoms severely impact daily functioning. Moreover, early detection facilitates participation in clinical trials investigating novel AD therapies, fostering hope for more effective future treatments. By advancing ensemble machine learning for enhanced early AD detection, this research holds the potential to transform diagnosis and empower those affected to confront this disease proactively when intervention is maximally impactful.

        ## Purpose of the project
        The purpose of this dissertation is to develop a machine learning model aimed at enhancing the early detection of AD. Alzheimer's disease is a significant neurodegenerative condition that impacts millions globally, with its prevalence expected to rise due to an aging population. Early detection is essential for improving patient outcomes and facilitating the development of targeted interventions. This research focuses on utilising advanced machine learning techniques, particularly stacking ensemble learning models, to construct a predictive model capable of identifying individuals at risk of AD by analysing multimodal datasets, including neuroimaging, genetic markers, and cognitive assessments.
        
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

    st.write("""
    <br>
    """, unsafe_allow_html=True)

    
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
