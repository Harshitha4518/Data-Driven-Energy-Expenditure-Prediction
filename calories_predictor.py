import os
import pickle
import warnings
import pandas as pd
import numpy as np
import customtkinter as tk
import matplotlib.pyplot as plt
import seaborn as sns

from tkinter import messagebox
from PIL import Image
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

tk.set_appearance_mode("dark")
tk.set_default_color_theme("blue")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "calories_model.pkl")
EDA_PATH = os.path.join(BASE_DIR, "eda_plots.png")
MODEL_COMP_PATH = os.path.join(BASE_DIR, "model_comparison.png")
FEATURE_IMPORTANCE_PATH = os.path.join(BASE_DIR, "feature_importance.png")
METRICS_PATH = os.path.join(BASE_DIR, "metrics_card.png")


def get_path(filename):
    return os.path.join(BASE_DIR, filename)


def read_csv(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Missing file: {file_path}")
    return pd.read_csv(file_path)


def separate_features_target(data, target_column):
    X = data.drop(columns=[target_column])
    y = data[target_column]
    return X, y


def validate_inputs(gender, age, height, weight, duration, heart_rate, body_temp):
    if gender not in ["male", "female"]:
        raise ValueError("Invalid gender")
    if not (18 <= age <= 100):
        raise ValueError("Age must be 18-100")
    if not (140 <= height <= 220):
        raise ValueError("Height must be 140-220 cm")
    if not (35 <= weight <= 200):
        raise ValueError("Weight must be 35-200 kg")
    if not (1 <= duration <= 300):
        raise ValueError("Duration must be 1-300 min")
    if not (60 <= heart_rate <= 220):
        raise ValueError("Heart rate must be 60-220 bpm")
    if not (36 <= body_temp <= 42):
        raise ValueError("Body temp must be 36-42 C")
    return True


def generate_complete_analytics():
    print("🔄 Generating complete analytics suite...")

    calories = read_csv(get_path("calories.csv"))
    exercise = read_csv(get_path("exercise.csv"))
    data = pd.merge(calories, exercise, on="User_ID")

    print("Dataset info:")
    print(f"Shape: {data.shape}")
    print(f"Nulls:\n{data.isnull().sum()}")
    print("\nDescribe:\n", data.describe())

    # -------------------- EDA --------------------
    plt.figure(figsize=(20, 5))

    plt.subplot(1, 4, 1)
    sns.histplot(data["Age"], kde=True, color="#4da6ff")
    plt.title("Age Distribution")

    plt.subplot(1, 4, 2)
    sns.histplot(data["Height"], kde=True, color="#ff9933")
    plt.title("Height Distribution")

    plt.subplot(1, 4, 3)
    sns.histplot(data["Weight"], kde=True, color="#66cc66")
    plt.title("Weight Distribution")

    plt.subplot(1, 4, 4)
    numeric_data = data.select_dtypes(include=[np.number]).drop(columns=["User_ID"], errors="ignore")
    sns.heatmap(numeric_data.corr(), annot=True, cmap="coolwarm", center=0, fmt=".2f")
    plt.title("Correlation Heatmap")

    plt.tight_layout()
    plt.savefig(EDA_PATH, dpi=300, bbox_inches="tight")
    plt.close()

    # -------------------- Data prep --------------------
    X, y = separate_features_target(data, "Calories")
    X = X.drop(columns=["User_ID"], errors="ignore")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    feature_names = ["Gender", "Age", "Height", "Weight", "Duration", "Heart_Rate", "Body_Temp"]

    preprocessor = ColumnTransformer([
        ("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), ["Gender"]),
        ("num", StandardScaler(), ["Age", "Height", "Weight", "Duration", "Heart_Rate", "Body_Temp"])
    ])

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "XGBoost": XGBRegressor(
            n_estimators=100,
            random_state=42,
            objective="reg:squarederror",
            eval_metric="rmse"
        )
    }

    results = {}
    pipelines = {}
    best_pipeline = None
    best_r2 = -999
    best_name = None

    # -------------------- Train models --------------------
    for name, model in models.items():
        pipe = Pipeline([
            ("prep", preprocessor),
            ("model", model)
        ])

        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        cv_r2 = cross_val_score(pipe, X_train, y_train, cv=5, scoring="r2").mean()

        results[name] = {
            "R2": r2,
            "MAE": mae,
            "CV_R2": cv_r2
        }

        pipelines[name] = pipe

        print(f"{name}: R2={r2:.4f}, MAE={mae:.2f}, CV_R2={cv_r2:.4f}")

        if r2 > best_r2:
            best_r2 = r2
            best_pipeline = pipe
            best_name = name

    # Save best model
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(best_pipeline, f)

    # -------------------- Model comparison chart --------------------
    plt.figure(figsize=(10, 6))
    names = list(results.keys())
    r2_scores = [results[name]["R2"] for name in names]

    bars = plt.bar(names, r2_scores, color=["#4da6ff", "#66cc66", "#ff9933"], alpha=0.85)
    plt.ylim(0.85, 1.0)
    plt.title("Model Performance Comparison (R² Score)", fontsize=16, fontweight="bold")
    plt.ylabel("R² Score", fontsize=12)
    plt.xlabel("Models", fontsize=12)

    for bar, score in zip(bars, r2_scores):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            score + 0.003,
            f"{score:.3f}",
            ha="center",
            fontweight="bold"
        )

    plt.tight_layout()
    plt.savefig(MODEL_COMP_PATH, dpi=300, bbox_inches="tight")
    plt.close()

    # -------------------- Feature importance --------------------
    if "XGBoost" in pipelines:
        xgb_pipe = pipelines["XGBoost"]
        xgb_model = xgb_pipe.named_steps["model"]

        importances = xgb_model.feature_importances_
        indices = np.argsort(importances)[::-1]
        sorted_features = np.array(feature_names)[indices]
        sorted_importances = importances[indices]

        plt.figure(figsize=(10, 6))
        plt.bar(range(len(sorted_importances)), sorted_importances, color="#ff6b6b", alpha=0.85)
        plt.xticks(range(len(sorted_features)), sorted_features, rotation=45)
        plt.title("XGBoost Feature Importance", fontsize=16, fontweight="bold")
        plt.ylabel("Importance Score", fontsize=12)
        plt.xlabel("Features", fontsize=12)
        plt.tight_layout()
        plt.savefig(FEATURE_IMPORTANCE_PATH, dpi=300, bbox_inches="tight")
        plt.close()

    # -------------------- Metrics card --------------------
    best_metrics = results[best_name]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.axis("off")

    textstr = f"""BEST MODEL: {best_name}

R² Score (Test): {best_metrics['R2']:.4f}
MAE (Test): {best_metrics['MAE']:.2f} kcal
CV R² (5-fold): {best_metrics['CV_R2']:.4f}
Dataset: 15,000 samples
Features: 7 (Gender + 6 biometrics)"""

    ax.text(
        0.1,
        0.5,
        textstr,
        transform=ax.transAxes,
        fontsize=16,
        verticalalignment="center",
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#2e3440", alpha=0.9)
    )

    plt.title("Model Performance Metrics", fontsize=18, fontweight="bold", pad=20)
    plt.tight_layout()
    plt.savefig(METRICS_PATH, dpi=300, bbox_inches="tight")
    plt.close()

    print("✅ Analytics suite generated successfully!")


class AnalyticsDashboard(tk.CTkToplevel):
    def __init__(self, parent):
        super().__init__(parent)
        self.title("Project Analytics Dashboard")
        self.geometry("1350x900")

        title = tk.CTkLabel(
            self,
            text="ML Analytics Dashboard",
            font=tk.CTkFont(size=30, weight="bold")
        )
        title.pack(pady=20)

        scroll = tk.CTkScrollableFrame(self, width=1250, height=780)
        scroll.pack(padx=20, pady=20, fill="both", expand=True)

        tk.CTkLabel(
            scroll,
            text="Exploratory Data Analysis",
            font=tk.CTkFont(size=22, weight="bold")
        ).pack(pady=10)

        if os.path.exists(EDA_PATH):
            self.eda_img = tk.CTkImage(
                light_image=Image.open(EDA_PATH),
                dark_image=Image.open(EDA_PATH),
                size=(1150, 280)
            )
            tk.CTkLabel(scroll, image=self.eda_img, text="").pack(pady=10)
            tk.CTkLabel(
                scroll,
                text="This section shows age, height, weight distributions and correlation between numerical features.",
                font=tk.CTkFont(size=14)
            ).pack()

        tk.CTkLabel(
            scroll,
            text="Model Comparison",
            font=tk.CTkFont(size=22, weight="bold")
        ).pack(pady=(30, 10))

        if os.path.exists(MODEL_COMP_PATH):
            self.model_img = tk.CTkImage(
                light_image=Image.open(MODEL_COMP_PATH),
                dark_image=Image.open(MODEL_COMP_PATH),
                size=(850, 500)
            )
            tk.CTkLabel(scroll, image=self.model_img, text="").pack(pady=10)
            tk.CTkLabel(
                scroll,
                text="This graph compares Linear Regression, Random Forest, and XGBoost using R² score.",
                font=tk.CTkFont(size=14)
            ).pack()

        tk.CTkLabel(
            scroll,
            text="Feature Importance",
            font=tk.CTkFont(size=22, weight="bold")
        ).pack(pady=(30, 10))

        if os.path.exists(FEATURE_IMPORTANCE_PATH):
            self.feature_img = tk.CTkImage(
                light_image=Image.open(FEATURE_IMPORTANCE_PATH),
                dark_image=Image.open(FEATURE_IMPORTANCE_PATH),
                size=(850, 500)
            )
            tk.CTkLabel(scroll, image=self.feature_img, text="").pack(pady=10)
            tk.CTkLabel(
                scroll,
                text="This graph explains which input features most influence calorie prediction in XGBoost.",
                font=tk.CTkFont(size=14)
            ).pack()

        tk.CTkLabel(
            scroll,
            text="Performance Metrics",
            font=tk.CTkFont(size=22, weight="bold")
        ).pack(pady=(30, 10))

        if os.path.exists(METRICS_PATH):
            self.metrics_img = tk.CTkImage(
                light_image=Image.open(METRICS_PATH),
                dark_image=Image.open(METRICS_PATH),
                size=(700, 400)
            )
            tk.CTkLabel(scroll, image=self.metrics_img, text="").pack(pady=10)


class CaloriesPredictorPro:
    def __init__(self):
        self.root = tk.CTk()
        self.root.title("Calories Burnt Predictor PRO")
        self.root.geometry("700x950")
        self.root.resizable(False, False)

        required_files = [
            MODEL_PATH,
            EDA_PATH,
            MODEL_COMP_PATH,
            FEATURE_IMPORTANCE_PATH,
            METRICS_PATH
        ]

        if any(not os.path.exists(path) for path in required_files):
            generate_complete_analytics()

        self.model = self.load_model()
        self.setup_ui()

    def load_model(self):
        with open(MODEL_PATH, "rb") as f:
            return pickle.load(f)

    def setup_ui(self):
        title = tk.CTkLabel(
            self.root,
            text="Calories Burnt Predictor",
            font=tk.CTkFont(size=32, weight="bold")
        )
        title.pack(pady=20)

        subtitle = tk.CTkLabel(
            self.root,
            text="Professional ML System with Analytics Dashboard",
            font=tk.CTkFont(size=15)
        )
        subtitle.pack(pady=(0, 20))

        input_frame = tk.CTkFrame(self.root)
        input_frame.pack(padx=20, pady=10, fill="x")

        tk.CTkLabel(
            input_frame,
            text="User Input",
            font=tk.CTkFont(size=20, weight="bold")
        ).pack(pady=15)

        fields = [
            ("Gender", "male", ["male", "female"]),
            ("Age (years)", "25", None),
            ("Height (cm)", "170", None),
            ("Weight (kg)", "70", None),
            ("Duration (min)", "30", None),
            ("Heart Rate (bpm)", "100", None),
            ("Body Temp (°C)", "37.5", None)
        ]

        self.entries = {}

        for label, default, options in fields:
            row = tk.CTkFrame(input_frame)
            row.pack(fill="x", padx=20, pady=8)

            tk.CTkLabel(
                row,
                text=label,
                width=150,
                font=tk.CTkFont(size=14)
            ).pack(side="left", padx=10)

            if options:
                widget = tk.CTkOptionMenu(row, values=options, width=220)
                widget.set(default)
            else:
                widget = tk.CTkEntry(row, width=220)
                widget.insert(0, default)

            widget.pack(side="right", padx=10)
            self.entries[label] = widget

        btn_frame = tk.CTkFrame(self.root)
        btn_frame.pack(pady=20)

        predict_btn = tk.CTkButton(
            btn_frame,
            text="Predict Calories",
            command=self.predict_calories,
            height=45,
            width=220,
            font=tk.CTkFont(size=17, weight="bold"),
            fg_color="#ff6b6b",
            hover_color="#e05555"
        )
        predict_btn.pack(side="left", padx=10)

        analytics_btn = tk.CTkButton(
            btn_frame,
            text="Show Analytics",
            command=self.show_analytics,
            height=45,
            width=220,
            font=tk.CTkFont(size=17, weight="bold"),
            fg_color="#4da6ff",
            hover_color="#357abd"
        )
        analytics_btn.pack(side="left", padx=10)

        self.result_label = tk.CTkLabel(
            self.root,
            text="Enter values and click Predict Calories",
            font=tk.CTkFont(size=26, weight="bold"),
            text_color="orange"
        )
        self.result_label.pack(pady=30)

        self.guide_label = tk.CTkLabel(
            self.root,
            text="Judge Demo Flow: Predict → Show Analytics → Explain Best Model + Feature Importance",
            font=tk.CTkFont(size=14),
            text_color="gray"
        )
        self.guide_label.pack()

    def predict_calories(self):
        try:
            gender = self.entries["Gender"].get()
            age = float(self.entries["Age (years)"].get())
            height = float(self.entries["Height (cm)"].get())
            weight = float(self.entries["Weight (kg)"].get())
            duration = float(self.entries["Duration (min)"].get())
            heart_rate = float(self.entries["Heart Rate (bpm)"].get())
            body_temp = float(self.entries["Body Temp (°C)"].get())

            validate_inputs(gender, age, height, weight, duration, heart_rate, body_temp)

            sample = pd.DataFrame({
                "Gender": [gender],
                "Age": [age],
                "Height": [height],
                "Weight": [weight],
                "Duration": [duration],
                "Heart_Rate": [heart_rate],
                "Body_Temp": [body_temp]
            })

            prediction = self.model.predict(sample)[0]

            self.result_label.configure(
                text=f"Predicted Calories Burnt: {prediction:.2f} kcal",
                text_color="lightgreen"
            )

        except Exception as e:
            messagebox.showerror("Input Error", str(e))
            self.result_label.configure(
                text="Invalid Input",
                text_color="red"
            )

    def show_analytics(self):
        AnalyticsDashboard(self.root)

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    print("🚀 Launching Calories Predictor PRO - Final Year Project")
    app = CaloriesPredictorPro()
    app.run()