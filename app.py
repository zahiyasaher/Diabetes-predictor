import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go

# Configure page settings
st.set_page_config(
    page_title="Diabetes Prediction AI",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        color: #1f2937;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #3b82f6, #8b5cf6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .sub-header {
        font-size: 1.5rem;
        color: #4b5563;
        text-align: center;
        margin-bottom: 3rem;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(31, 38, 135, 0.37);
    }
    
    .feature-importance {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    .prediction-box {
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin: 2rem 0;
        font-size: 1.2rem;
        font-weight: 600;
    }
    
    .positive-prediction {
        background: linear-gradient(135deg, #ff6b6b, #ee5a52);
        color: white;
        box-shadow: 0 8px 32px rgba(255, 107, 107, 0.3);
    }
    
    .negative-prediction {
        background: linear-gradient(135deg, #51cf66, #40c057);
        color: white;
        box-shadow: 0 8px 32px rgba(81, 207, 102, 0.3);
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #f8fafc 0%, #e2e8f0 100%);
    }
    
    .stSlider > div > div > div > div {
        background: linear-gradient(90deg, #3b82f6, #8b5cf6);
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_and_prepare_data():
    """Load and prepare the diabetes dataset"""
    try:
        diabetes_df = pd.read_csv('diabetes.csv')
        return diabetes_df
    except FileNotFoundError:
        st.error("❌ Dataset 'diabetes.csv' not found. Please ensure the file is in the correct directory.")
        return None

@st.cache_resource
def train_model(diabetes_df):
    """Train the SVM model"""
    # Group data by outcome
    diabetes_mean_df = diabetes_df.groupby('Outcome').mean()
    
    # Split features and target
    X = diabetes_df.drop('Outcome', axis=1)
    y = diabetes_df['Outcome']
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=1, stratify=y
    )
    
    # Train model
    model = svm.SVC(kernel='linear', probability=True)
    model.fit(X_train, y_train)
    
    # Calculate accuracies
    train_acc = accuracy_score(y_train, model.predict(X_train))
    test_acc = accuracy_score(y_test, model.predict(X_test))
    
    return model, scaler, train_acc, test_acc, diabetes_mean_df, diabetes_df





def main():
    # Header
    st.markdown('<h1 class="main-header">🩺 Diabetes Prediction AI</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Advanced Machine Learning for Diabetes Risk Assessment</p>', unsafe_allow_html=True)
    
    # Load data and train model
    diabetes_df = load_and_prepare_data()
    if diabetes_df is None:
        return
    
    model, scaler, train_acc, test_acc, diabetes_mean_df, df = train_model(diabetes_df)

      # Sidebar for input features
    st.sidebar.markdown("## 📊 Patient Information")
    st.sidebar.markdown("---")
    
    # Input sliders with better descriptions and icons
    preg = st.sidebar.slider('🤰 Number of Pregnancies', 0, 17, 3, 
                            help="Number of times pregnant")
    
    glucose = st.sidebar.slider('🍯 Glucose Level (mg/dL)', 0, 199, 117,
                               help="Plasma glucose concentration after 2hr oral glucose tolerance test")
    
    bp = st.sidebar.slider('💓 Blood Pressure (mmHg)', 0, 122, 72,
                          help="Diastolic blood pressure")
    
    skinthickness = st.sidebar.slider('📏 Skin Thickness (mm)', 0, 99, 23,
                                     help="Triceps skin fold thickness")
    
    insulin = st.sidebar.slider('💉 Insulin Level (μU/mL)', 0, 846, 30,
                               help="2-Hour serum insulin")
    
    bmi = st.sidebar.slider('⚖️ Body Mass Index', 0.0, 67.1, 32.0,
                           help="Body mass index (weight in kg/(height in m)^2)")
    
    dpf = st.sidebar.slider('🧬 Diabetes Pedigree Function', 0.078, 2.42, 0.3725, 0.001,
                           help="Diabetes pedigree function (genetic risk factor)")
    
    age = st.sidebar.slider('🎂 Age (years)', 21, 81, 29,
                           help="Age in years")
