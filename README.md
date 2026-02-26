# 🏠 Housing Price Analysis & Prediction

A production-grade **Streamlit** web application for comprehensive housing price analysis and prediction. This app provides an end-to-end ML pipeline — from exploratory data analysis to model building, SHAP explainability, and an AI-powered chatbot.

---

## ✨ Features

| # | Feature | Description |
|---|---------|-------------|
| 1 | **Data Understanding** | Dataset overview, shape, types, missing values, and summary statistics |
| 2 | **Advanced EDA** | Univariate, bivariate, and multivariate analysis with matplotlib visualizations |
| 3 | **Data Preprocessing** | Log-transform, one-hot encoding, standard scaling, and 80/20 train-test split |
| 4 | **Model Building** | 6 regressors with GridSearchCV hyperparameter tuning |
| 5 | **Model Evaluation** | Comparison table, actual-vs-predicted plots, and residual distributions |
| 6 | **Feature Importance** | Tree-based feature importance ranking with bar charts |
| 7 | **SHAP Explainability** | Summary, bar, and force plots for model interpretability |
| 8 | **Prediction Interface** | Interactive sidebar inputs for real-time price prediction |
| 9 | **Business Insights** | Auto-generated investment & builder recommendations |
| 10 | **Downloadable Reports** | Export model comparison and feature importance as CSV |
| 11 | **Gemini AI Chatbot** | Data-analysis Q&A powered by Google Gemini |

---

## 🤖 Models Used

- **Linear Regression**
- **Ridge Regression** (GridSearchCV)
- **Lasso Regression** (GridSearchCV)
- **Random Forest Regressor** (GridSearchCV)
- **Gradient Boosting Regressor** (GridSearchCV)
- **XGBoost Regressor** (GridSearchCV)

---

## 🛠️ Tech Stack

- **Frontend:** Streamlit
- **Data Processing:** Pandas, NumPy, SciPy
- **Visualization:** Matplotlib
- **Machine Learning:** Scikit-learn, XGBoost
- **Explainability:** SHAP
- **AI Chatbot:** Google Generative AI (Gemini)
- **Statistical Analysis:** Statsmodels (VIF)

---

## 📦 Installation

### Prerequisites

- Python 3.9 or higher

### Steps

1. **Clone the repository**
   ```bash
   git clone https://github.com/akshay-kumar-06/HousePricePrediction
   cd HousePricePrediction
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   ```

3. **Activate the virtual environment**
   ```bash
   # Windows
   venv\Scripts\activate

   # macOS / Linux
   source venv/bin/activate
   ```

4. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

---

## 🚀 Usage

Run the Streamlit application:

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

### Using the AI Chatbot

To use the Gemini-powered chatbot, enter your **Google Gemini API Key** in the sidebar. You can get one from [Google AI Studio](https://aistudio.google.com/apikey).

---

## 📊 Dataset

The application uses the **Housing Price dataset** (`Assignment_1_MLR_CF_Housing_Price.csv`) which contains the following features:

| Feature | Description |
|---------|-------------|
| `price` | Price of the house (target variable) |
| `area` | Area of the house in sq ft |
| `bedrooms` | Number of bedrooms |
| `bathrooms` | Number of bathrooms |
| `stories` | Number of stories |
| `mainroad` | Whether connected to the main road (yes/no) |
| `guestroom` | Whether it has a guestroom (yes/no) |
| `basement` | Whether it has a basement (yes/no) |
| `hotwaterheating` | Whether it has hot water heating (yes/no) |
| `airconditioning` | Whether it has air conditioning (yes/no) |
| `parking` | Number of parking spots |
| `prefarea` | Whether in a preferred area (yes/no) |
| `furnishingstatus` | Furnishing status (furnished / semi-furnished / unfurnished) |

---

## 📁 Project Structure

```
Roman Technology/
├── app.py                                    # Main Streamlit application
├── Assignment_1_MLR_CF_Housing_Price.csv     # Housing price dataset
├── requirements.txt                          # Python dependencies
├── README.md                                 # Project documentation
└── venv/                                     # Virtual environment
```

---

## 📸 Screenshots

Once the app is running, you'll see:

- 📊 **EDA Tab** — Histograms, boxplots, scatter plots, and correlation heatmaps
- 📈 **Model Evaluation Tab** — Side-by-side model comparison with visual diagnostics
- 🔍 **SHAP Tab** — Feature-level model explanations
- 🎯 **Prediction Tab** — Predict housing prices with interactive inputs
- 💼 **Business Insights Tab** — Data-driven recommendations
- 🤖 **AI Chatbot Tab** — Ask questions about the data and models

---

## 📄 License

This project is for educational and academic purposes.

---

## 🙏 Acknowledgements

- [Streamlit](https://streamlit.io/) for the web framework
- [SHAP](https://github.com/slundberg/shap) for model explainability
- [XGBoost](https://xgboost.readthedocs.io/) for gradient boosting
- [Google Gemini](https://ai.google.dev/) for the AI chatbot
