import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(
    page_title="Teen Smartphone Addiction",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for full clickable sidebar
st.markdown("""
    <style>
        .sidebar .sidebar-content {
            padding: 10px;
        }
        .sidebar .sidebar-content div {
            padding: 8px;
            font-size: 16px;
            border-radius: 5px;
        }
        .sidebar .sidebar-content div:hover {
            background-color: #f0f2f6;
            cursor: pointer;
        }
    </style>
""", unsafe_allow_html=True)

# -------------------------------
# Sidebar Menu
# -------------------------------
st.sidebar.title("🔹My Streamlit Dashboard")
menu = st.sidebar.radio(
    "Go to",
    ["📄 Preview Dataset", "📊 Model Comparison", "🔮 Prediction", "ℹ️ About"],
    index=2
)

# -------------------------------
# Load Dataset
# -------------------------------
df = pd.read_csv("teen_phone_addiction_dataset.csv")
drop_cols = ["ID", "Name", "Location", "School_Grade", "Phone_Usage_Purpose"]
df = df.drop(columns=drop_cols, errors="ignore")
df["Addiction_Status"] = df["Addiction_Level"].apply(lambda x: "Addicted" if x >= 7 else "Not Addicted")

X = df.drop(["Addiction_Level", "Addiction_Status"], axis=1)
y = df["Addiction_Status"]
X = pd.get_dummies(X, drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -------------------------------
# Train Models
# -------------------------------
classifiers = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "Support Vector Machine": SVC(probability=True),
    "Naive Bayes": GaussianNB(),
    "K-Nearest Neighbors": KNeighborsClassifier()
}

results = []
trained_models = {}
for name, clf in classifiers.items():
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", clf)
    ])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, pos_label="Addicted", zero_division=0)
    rec = recall_score(y_test, y_pred, pos_label="Addicted", zero_division=0)
    f1 = f1_score(y_test, y_pred, pos_label="Addicted", zero_division=0)

    results.append([name, acc, prec, rec, f1])
    trained_models[name] = pipe

results_df = pd.DataFrame(results, columns=["Algorithm", "Accuracy", "Precision", "Recall", "F1 Score"])
best_model_name = results_df.sort_values(by="Accuracy", ascending=False).iloc[0]["Algorithm"]
best_model = trained_models[best_model_name]

# -------------------------------
# Pages Based on Sidebar Menu
# -------------------------------
if menu == "📄 Preview Dataset":
    st.title("📊 Dataset Preview")
    st.write("Preview of the Teen Smartphone Addiction Dataset")
    st.dataframe(df.head())
    st.write(f"Dataset contains {df.shape[0]} rows and {df.shape[1]} columns.")
    st.write("**Column names:**", list(df.columns))

elif menu == "📊 Model Comparison":
    st.title("🤖 Model Comparison")
    col1, col2 = st.columns([1, 2])
    with col1:
        st.dataframe(results_df)
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        col_m1.metric("Best Accuracy", f"{results_df['Accuracy'].max():.2f}")
        col_m2.metric("Best Precision", f"{results_df['Precision'].max():.2f}")
        col_m3.metric("Best Recall", f"{results_df['Recall'].max():.2f}")
        col_m4.metric("Best F1 Score", f"{results_df['F1 Score'].max():.2f}")
    with col2:
        fig = px.bar(results_df, x="Algorithm", y="Accuracy", color="Algorithm",
                     title="Algorithm Accuracy Comparison", text_auto=".2f")
        st.plotly_chart(fig, use_container_width=True)
    st.success(f"🏆 Best Model Selected: {best_model_name}")

elif menu == "🔮 Prediction":
    st.title("🔮 Predict Addiction")
    st.write("Fill the form below to predict if a person is addicted to smartphones.")

    tab1, tab2, tab3 = st.tabs(["📱 Usage Behavior", "🧠 Psychological Factors", "👨‍👩‍👦 Lifestyle & Academics"])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            daily_usage = st.slider("Daily Usage Hours", 0.0, 12.0, 4.0)
            social_media = st.slider("Time on Social Media (hrs)", 0.0, 8.0, 2.0)
            gaming = st.slider("Time on Gaming (hrs)", 0.0, 8.0, 1.0)
        with col2:
            education = st.slider("Time on Education (hrs)", 0.0, 8.0, 1.0)
            weekend = st.slider("Weekend Usage Hours", 0.0, 15.0, 5.0)
            checks = st.slider("Phone Checks Per Day", 0, 200, 50)

    with tab2:
        col3, col4 = st.columns(2)
        with col3:
            anxiety = st.slider("Anxiety Level", 0, 10, 5)
            depression = st.slider("Depression Level", 0, 10, 5)
        with col4:
            self_esteem = st.slider("Self Esteem", 0, 10, 5)

    with tab3:
        col5, col6 = st.columns(2)
        with col5:
            age = st.number_input("Age", min_value=10, max_value=25, value=15)
            gender = st.selectbox("Gender", ["Male", "Female", "Other"])
            sleep = st.slider("Sleep Hours", 0.0, 12.0, 7.0)
        with col6:
            academic = st.slider("Academic Performance", 0, 100, 75)
            social = st.slider("Social Interactions", 0, 10, 5)
            exercise = st.slider("Exercise Hours", 0.0, 5.0, 1.0)
            parental = st.slider("Parental Control", 0, 10, 5)
            screen_bed = st.slider("Screen Time Before Bed (hrs)", 0.0, 5.0, 1.0)
            family = st.slider("Family Communication (hrs)", 0, 15, 5)

    input_data = pd.DataFrame([[age, daily_usage, sleep, academic, social, exercise,
                                anxiety, depression, self_esteem, parental, screen_bed,
                                checks, (social_media+gaming+education), social_media, gaming, education,
                                family, weekend, gender]],
                              columns=["Age", "Daily_Usage_Hours", "Sleep_Hours", "Academic_Performance",
                                       "Social_Interactions", "Exercise_Hours", "Anxiety_Level", "Depression_Level",
                                       "Self_Esteem", "Parental_Control", "Screen_Time_Before_Bed",
                                       "Phone_Checks_Per_Day", "Apps_Used_Daily", "Time_on_Social_Media",
                                       "Time_on_Gaming", "Time_on_Education", "Family_Communication",
                                       "Weekend_Usage_Hours", "Gender"])
    input_data = pd.get_dummies(input_data, drop_first=True)
    input_data = input_data.reindex(columns=X.columns, fill_value=0)

    if st.button("Predict Addiction"):
        proba = best_model.predict_proba(input_data)[0][1]
        prediction = best_model.predict(input_data)[0]

        st.progress(int(proba*100))
        if prediction == "Addicted":
            st.error("🚨 The person is predicted to be **ADDICTED** to smartphone usage.")
            st.info("⚠️ Recommendation: Reduce screen time, increase physical activities, and practice mindfulness.")
        else:
            st.success("✅ The person is predicted to be **NOT ADDICTED**.")

elif menu == "ℹ️ About":
    st.title("ℹ️ About This App")
    st.markdown("""
    **Assignment No.:** 5  
    **Name:** Vivek Sainath Bhange  
    **PRN No.:** 22310090  
    **Roll No.:** 322003  
    **Branch:** Computer Engineering  
    **Subject:** Artificial Intelligence  
    """)
    st.markdown("""
    This app predicts whether a teenager is addicted to smartphone usage based on several factors.
    
    **Features:**
    - Dataset preview
    - Comparison of different ML models
    - Addiction prediction
    - Recommendations
    
    Built with Streamlit, Scikit-learn & Plotly.
    """)
