# Employee-Attrition-Prediction-
A Streamlit web application that predicts employee attrition risk using machine learning.

Project Overview

This application analyzes employee data and predicts the likelihood of attrition using a Random Forest classifier. It provides both batch analysis of existing data and interactive individual employee predictions.

 Features

Data Analysis: Upload and explore employee datasets
Batch Predictions: Analyze attrition patterns across entire departments
Individual Predictions: Interactive form for predicting specific employee attrition risk
Feature Importance: Identify key factors influencing employee turnover
User-Friendly Interface: Clean, intuitive Streamlit web interface

Installation

1. Clone the repository:
   git clone https://github.com/yourusername/employee-attrition-prediction.git
   cd employee-attrition-prediction

2. Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

3. Install dependencies:
pip install -r requirements.txt
streamlit>=1.28.0
pandas>=1.5.0
numpy>=1.21.0
scikit-learn>=1.0.0
matplotlib>=3.5.0

Usage

1. Run the application: streamlit run app.py 
2. Upload your data:
    * The app will open in your browser at http://localhost:8501
    * Upload your employeeattri.csv file using the file uploader

1. View predictions:
    * See overall attrition statistics
    * Check model accuracy and feature importance
	◦	Use the interactive form to predict individual employee attrition risk


Project Structure
employee-attrition-prediction/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies        
└── data/
    └── employeeattri.csv  

 How It Works
1. Data Preprocessing:
    * Converts categorical variables to numerical
    * Handles missing values
    * Selects relevant features for prediction
2. Machine Learning:
    * Uses Random Forest classifier
    * 70-30 train-test split
    * Evaluates model accuracy
3. Predictions:
    * Batch predictions on test data
    * Individual predictions via user input
	  * Probability scores for attrition risk

Model Performance
* Typical accuracy: 85-90%
* Top predictive features: Monthly Income, Overtime, Job Satisfaction, Years at Company
* Real-time risk assessment with probability scores   Contributing
1. Fork the repository
2. Create a feature branch (git checkout -b feature/AmazingFeature)
3. Commit your changes (git commit -m 'Add some AmazingFeature')
4. Push to the branch (git push origin feature/AmazingFeature)
5. Open a Pull Request



Author
* Sailee Singh Maharjan 

 Acknowledgments
* IBM for the HR Analytics Dataset
* Streamlit for the web framework
* Scikit-learn for machine learning capabilities
