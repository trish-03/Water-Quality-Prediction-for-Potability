import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load the saved model
model_path = 'E:\\Boot_camp\\KEC_Bootcamp\\water_potability_analysis\\ML_MODEL\\random_forest_model.pkl'
model = joblib.load(model_path)

# Extract feature names from the model if they are available
try:
    expected_columns = model.feature_names_in_
except AttributeError:
    st.error("The model does not contain feature names. Please ensure the model is trained with feature names.")

def main():
    # Set the title of the web app
    st.title('Water Potability Prediction')

    # Add a description
    st.write('Enter water quality parameters to predict potability.')

    # Create columns for layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader('Water Quality Parameters')

        # Add input fields for features
        ph = st.slider('pH Value', 0.0, 14.0, 7.0)
        hardness = st.slider('Hardness (mg/L)', 0.0, 500.0, 200.0)
        solids = st.slider('Solids (ppm)', 0.0, 50000.0, 20000.0)
        chloramines = st.slider('Chloramines (ppm)', 0.0, 10.0, 5.0)
        sulfate = st.slider('Sulfate (mg/L)', 0.0, 500.0, 250.0)
        conductivity = st.slider('Conductivity (µS/cm)', 0.0, 1000.0, 500.0)
        organic_carbon = st.slider('Organic Carbon (ppm)', 0.0, 30.0, 15.0)
        trihalomethanes = st.slider('Trihalomethanes (µg/L)', 0.0, 120.0, 60.0)
        turbidity = st.slider('Turbidity (NTU)', 0.0, 10.0, 5.0)

    # Prepare input data as a DataFrame
    input_data = pd.DataFrame({
        'ph': [ph],
        'Hardness': [hardness],
        'Solids': [solids],
        'Chloramines': [chloramines],
        'Sulfate': [sulfate],
        'Conductivity': [conductivity],
        'Organic_carbon': [organic_carbon],
        'Trihalomethanes': [trihalomethanes],
        'Turbidity': [turbidity]
    })

    # Ensure columns are in the same order as during model training
    input_data = input_data[expected_columns]

    # Prediction and results section
    with col2:
        st.subheader('Prediction')
        if st.button('Predict'):
            prediction = model.predict(input_data)
            probability = model.predict_proba(input_data)[0][1]
            
            st.write(f'Prediction: {"Potable" if prediction[0] == 1 else "Not Potable"}')
            st.write(f'Probability of Potability: {probability:.2f}')

            # Plotting
            fig, axes = plt.subplots(3, 1, figsize=(8, 16))

            # Plot Potable/Not Potable probability
            sns.barplot(x=['Not Potable', 'Potable'], y=[1 - probability, probability], ax=axes[0], palette=['red', 'green'])
            axes[0].set_title('Potability Probability')
            axes[0].set_ylabel('Probability')

            # Plot parameter distribution
            sns.histplot(input_data.values.flatten(), kde=True, ax=axes[1])
            axes[1].set_title('Parameter Distribution')

            # Plot Potable/Not Potable pie chart
            axes[2].pie([1 - probability, probability], labels=['Not Potable', 'Potable'], autopct='%1.1f%%', colors=['red', 'green'])
            axes[2].set_title('Potable/Not Potable Pie Chart')

            # Display the plots
            st.pyplot(fig)

            # Provide recommendations
            if prediction[0] == 1:
                st.success("The water is likely potable. It meets the quality standards for drinking.")
            else:
                st.error("The water is likely not potable. It does not meet the quality standards for drinking.")

if __name__ == '__main__':
    main()
