import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

# ---------------- Load Dataset ----------------
dataset = pd.read_csv("job_salary_prediction.csv")

# Impute numeric
imputer = SimpleImputer(strategy="mean")
dataset["age"] = imputer.fit_transform(dataset[["age"]])[:, 0]
dataset["experience"] = imputer.fit_transform(dataset[["experience"]])[:, 0]
dataset["salary"] = imputer.fit_transform(dataset[["salary"]])[:, 0]

# Impute categorical
cat_imputer = SimpleImputer(strategy="most_frequent")
dataset["gender"] = cat_imputer.fit_transform(dataset[["gender"]])[:, 0]
dataset["education_level"] = cat_imputer.fit_transform(dataset[["education_level"]])[
    :, 0
]
dataset["job_title"] = cat_imputer.fit_transform(dataset[["job_title"]])[:, 0]

# One-hot encode
dataset = pd.get_dummies(dataset, columns=["gender", "education_level", "job_title"])

# Features & target
x = dataset.drop("salary", axis=1)
y = dataset["salary"]

# Split & scale
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.1, random_state=90
)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(x_train)
X_test_scaled = scaler.transform(x_test)

# Model
model = XGBRegressor(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=90,
)
model.fit(X_train_scaled, y_train)


# Evaluation
y_pred = model.predict(X_test_scaled)
model_accuracy = model.score(X_test_scaled, y_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

# Save plot
plt.scatter(y_test, y_pred, alpha=0.7)
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    color="red",
    linestyle="--",
)
plt.xlabel("Actual Salary")
plt.ylabel("Predicted Salary")
plt.title("Actual vs Predicted Salary")
plt.grid(True)
plt.tight_layout()
plt.savefig("scatter_plot.jpg")
plt.close()

# Extract categories for dropdown
gender_options = ["Male", "Female", "Other"]
education_options = ["High School", "Diploma", "Bachelor", "Master", "PhD"]

job_options = [
    col.replace("job_title_", "") for col in x.columns if "job_title_" in col
]


# ---------------- Prediction Function ----------------
def predict_salary(age, gender, education_level, job_title, experience):
    input_dict = {col: 0 for col in x.columns}
    input_dict["age"] = int(age)
    input_dict["experience"] = experience

    # Encode one-hot manually
    gender_col = f"gender_{gender}"
    edu_col = f"education_level_{education_level}"
    job_col = f"job_title_{job_title}"

    for col in [gender_col, edu_col, job_col]:
        if col in input_dict:
            input_dict[col] = 1

    input_df = pd.DataFrame([input_dict])
    input_scaled = scaler.transform(input_df)
    pred = model.predict(input_scaled)[0]

    return (
        f"₹ {pred:.2f}",
        f"{model_accuracy*100:.2f}%",
        round(rmse, 2),
        round(r2, 2),
        "scatter_plot.jpg",
    )


# ---------------- Gradio UI ----------------
with gr.Blocks(title="Salary Prediction") as demo:
    gr.Markdown("# Job Salary Prediction using XGBoost")
    gr.Markdown("# Created by MOHD ALTAMASH")

    with gr.Row():
        age = gr.Slider(18, 75, value=25, label="Age", step=1)
        experience = gr.Slider(0, 40, step=1, value=1, label="Experience (Years)")

    with gr.Row():
        gender = gr.Radio(gender_options, label="Gender", value="Male")
        education = gr.Dropdown(education_options, label="Education Level")
        job = gr.Dropdown(job_options, label="Job Title")

    predict_btn = gr.Button(" Predict Salary")

    with gr.Row():
        output_salary = gr.Label(label="Predicted Salary ₹")
        output_acc = gr.Label(label="Model Accuracy")
        output_rmse = gr.Label(label="RMSE")
        output_r2 = gr.Label(label="R² Score")

    plot_output = gr.Image(type="filepath", label="Actual vs Predicted Plot")

    predict_btn.click(
        fn=predict_salary,
        inputs=[age, gender, education, job, experience],
        outputs=[output_salary, output_acc, output_rmse, output_r2, plot_output],
    )

demo.launch()
