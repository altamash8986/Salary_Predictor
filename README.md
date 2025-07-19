# Job Salary Prediction Web App (XGBoost + Gradio)

A complete end-to-end machine learning project that predicts job salaries based on candidate profile inputs. The solution leverages powerful preprocessing, feature engineering, and a robust **XGBoost Regression model**, presented through an interactive **Gradio GUI** for real-time predictions.

---

##  Problem Statement

In a data-driven hiring ecosystem, estimating a candidate’s expected salary based on attributes like education level, job title, and work experience can aid:
- HR professionals in decision-making,
- Job seekers in salary benchmarking,
- Analysts in building compensation models.

This project aims to build a **regression model** that accurately predicts salaries and deploys it as a web-based application using **Gradio**.

---

##  Dataset Information

- **Filename**: `job_salary_prediction.csv`
- **Type**: Synthetic or collected structured dataset
- **Target Variable**: `salary`
- **Total Columns**: 6

### Features:

| Feature           | Type         | Description                                  |
|-------------------|--------------|----------------------------------------------|
| `age`             | Numerical    | Candidate's age                              |
| `gender`          | Categorical  | Gender (Male, Female, Other)                 |
| `education_level` | Categorical  | Highest education (High School to PhD)       |
| `job_title`       | Categorical  | Current job role                             |
| `experience`      | Numerical    | Total years of work experience               |
| `salary`          | Target       | Actual salary in INR                         |

---

##  Tech Stack / Tools

- **Language**: Python
- **Libraries**:
  - `pandas`, `numpy`, `matplotlib` – Data analysis & visualization
  - `scikit-learn` – Preprocessing & evaluation
  - `xgboost` – Powerful regression model
  - `gradio` – User interface for live prediction
- **ML Concepts**:
  - Missing value imputation
  - One-hot encoding
  - Feature scaling
  - Model evaluation (R², RMSE)
  - Deployment (Gradio UI)

---

##  Model Pipeline

### 1.  Data Preprocessing:
- **Impute missing values**:
  - Median for numerical
  - Most frequent for categorical
- **One-hot encode** categorical columns (`gender`, `education_level`, `job_title`)
- **Scale features** using `StandardScaler`

### 2.  Model Training:
- **Algorithm**: `XGBRegressor` with custom hyperparameters
- **Train-test split**: 90% train / 10% test
- **Hyperparameters**:
  - `n_estimators=200`
  - `max_depth=5`
  - `learning_rate=0.05`
  - `subsample=0.8`, `colsample_bytree=0.8`

### 3.  Model Evaluation:
| Metric | Description                        | Value         |
|--------|------------------------------------|---------------|
| `R²`   | Variance explained by the model    | **0.92**      |
| `RMSE` | Average prediction error (in ₹)    | **13,800 ₹**  |
| `MAE`  | Mean Absolute Error                | ~10,400 ₹     |
| `Accuracy` | Based on R² Score              | **91.80%**    |

### 4.  Visualization:
- **Scatter plot** of `Actual vs Predicted` salary
- **Red dotted line** shows ideal predictions

---

##  Gradio Interface

###  Inputs:
- Age (Slider: 18–75)
- Gender (Radio)
- Education Level (Dropdown)
- Job Title (Dropdown)
- Experience (Slider: 0–40 years)

###  Outputs:
- **Predicted Salary**
- **Model Accuracy (%)**
- **RMSE**
- **R² Score**
- **Scatter Plot** (Actual vs Predicted)

---

## Live Demo

> You can deploy this project on:
- [Hugging Face Spaces](https://huggingface.co/spaces)
  
---

##  Installation & Running

```bash
# Step 1: Clone the repository
git clone https://github.com/yourusername/job-salary-prediction.git
cd job-salary-prediction

# Step 2: Install dependencies
pip install -r requirements.txt

# Step 3: Run the app
python job_salary_prediction.py
