# Water Quality Prediction for Potability

This project implements a water quality predictor to determine potability using machine learning techniques in Python. It includes preprocessing steps, model training, evaluation, and prediction functionalities.

## Dataset

The dataset used in this project contains the following columns:

| Column          | Non-Null Count | Dtype   |
| --------------- | -------------- | ------- |
| ph              | 2785           | float64 |
| Hardness        | 3276           | float64 |
| Solids          | 3276           | float64 |
| Chloramines     | 3276           | float64 |
| Sulfate         | 2495           | float64 |
| Conductivity    | 3276           | float64 |
| Organic_carbon  | 3276           | float64 |
| Trihalomethanes | 3114           | float64 |
| Turbidity       | 3276           | float64 |
| Potability      | 3276           | int64   |

This project will guide you through the steps needed to preprocess, train, evaluate, and make predictions using this data.

## Features

- **Data Preprocessing**: Clean and prepare the dataset for machine learning.
- **Model Training**: Train machine learning models using various algorithms ( Decision Tree classifier, RandomForestClassifier, KNN classifier, Gaussian Naive Bayes classifier, logistic regression model )
- **Evaluation**: Evaluate model performance using metrics like accuracy, precision, and recall.
- **Prediction**: Make predictions on new data using trained models.

  the Predictor is deployed on app.py, and the remaining data analysis is done in Water Analysis.ipynb.

## Installation

1. Clone this repository:

   ```
   git clone https://github.com/trish-03/Water-Quality-Prediction-for-Potability.git
   ```

2. Navigate to the project directory:

   ```
   cd Water-Quality-Prediction-for-Potability
   ```

3. Install dependencies:
   ```
   python -m pip install -r requirements.txt
   ```

## to run the predictor, follow these steps:

### The model has already been trained. Run the application script .i.e 'app.py' to start using the prediction model:

```
python -m streamlit run app.py
```

After execution, open your web browser and go to `http://127.0.0.1:8501` to access the application.

### If you want to train the model on your own:

1. **Install anaconda**: install anaconda from online. Once downloaded, jupyter notebook is installed automatically.

2. **Open the Anaconda prompt**: Navigate to the directory where `Water Analysis.ipynb` is located using your command line or terminal.

3. **Start Jupyter Notebook**: Run the following command to start the Jupyter Notebook server:
   ```
   jupyter notebook
   ```
   Then inside jupyter notebook, open `Water Analysis.ipynb`.
