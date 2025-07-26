# Diabetes Predictor - using Machine Learning

Diabetes is a chronic health condition affecting millions worldwide. Early detection and management are crucial for preventing complications and improving patient outcomes. This project leverages the power of machine learning to predict the likelihood of diabetes in individuals based on various health parameters such as glucose levels, BMI, age, and more.

Using a dataset containing historical patient information, advanced machine learning algorithms are trained to analyze patterns and identify predictive features associated with diabetes. The resulting model can accurately classify individuals into diabetic or non-diabetic categories, providing valuable insights for healthcare practitioners and empowering individuals to take proactive measures for their health.

By harnessing the capabilities of machine learning, this project aims to enhance diabetes diagnosis, facilitate early intervention, and ultimately contribute to better healthcare outcomes for individuals at risk of this prevalent disease.

<br>

➲ **Used Technologies:** <br>
Python - Streamlit - JavaScript - CSS - Python Libraries (pandas - numpy - matplotlib - seaborn - sklearn) - ML Algorithms (Logistic Regression - Support Vector Machine - Random Forest Classifier - Gradient Boosting Classifier)
<br><br>
➲ **Notebook (ML Code):** <a href="https://www.kaggle.com/code/ahmedmaheralgohary/diabetes-prediction" target="_blank">kaggle.com/code/ahmedmaheralgohary/diabetes-prediction</a>
<br>

## 🚀 Deployment Options

### 1. Streamlit Cloud (Recommended)

1. Push your code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub account
4. Select your repository and main branch
5. Deploy! Your app will be live at `https://your-app-name.streamlit.app`

### 2. Heroku

1. Create a `Procfile` with: `web: streamlit run app.py --server.port=$PORT --server.address=0.0.0.0`
2. Create a `setup.sh` file for configuration
3. Deploy to Heroku using Git

### 3. Railway

1. Connect your GitHub repository
2. Railway will automatically detect it's a Streamlit app
3. Deploy with one click

### 4. Local Development

```bash
pip install -r requirements.txt
streamlit run app.py
```

## 📁 Project Structure

```
Diabetes Prediction App/
├── app.py                    # Main Streamlit application
├── Diabetes-Prediction-ML-Model.sav  # Trained ML model
├── requirements.txt          # Python dependencies
├── style.css                # Custom styling
└── README.md               # Project documentation
```
