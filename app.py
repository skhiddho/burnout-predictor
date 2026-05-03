import streamlit as st
import pandas as pd
import joblib
import numpy as np

# 1. Load the trained model
model = joblib.load('burnout_model.pkl')

# 2. App Header
st.title("Developer Burnout Predictor")
st.write("Enter your daily habits below to see your burnout risk level.")

# 3. Sidebar for User Inputs
st.sidebar.header("User Input Metrics")

def user_input_features():
    age = st.sidebar.slider('Age', 18, 60, 25)
    exp = st.sidebar.slider('Years of Experience', 0, 40, 5)
    work_hrs = st.sidebar.slider('Daily Work Hours', 1, 16, 8)
    sleep_hrs = st.sidebar.slider('Sleep Hours', 1, 12, 7)
    caffeine = st.sidebar.slider('Caffeine Intake (Cups)', 0, 10, 2)
    bugs = st.sidebar.number_input('Bugs encountered per day', 0, 50, 5)
    commits = st.sidebar.number_input('Commits per day', 0, 50, 10)
    meetings = st.sidebar.slider('Meetings per day', 0, 15, 3)
    screen_time = st.sidebar.slider('Screen Time (Hours)', 1, 20, 10)
    exercise = st.sidebar.slider('Exercise Hours', 0, 5, 1)

    data = {
        'age': age,
        'experience_years': exp,
        'daily_work_hours': work_hrs,
        'sleep_hours': sleep_hrs,
        'caffeine_intake': caffeine,
        'bugs_per_day': bugs,
        'commits_per_day': commits,
        'meetings_per_day': meetings,
        'screen_time': screen_time,
        'exercise_hours': exercise,
    }
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

# 4. Applying Feature Engineering (Must match the Training logic!)
input_df['work_recovery_ratio'] = input_df['daily_work_hours'] / (input_df['sleep_hours'] + 1)
input_df['meeting_load'] = input_df['meetings_per_day'] * input_df['daily_work_hours']

# 5. Prediction
prediction = model.predict(input_df)
prediction_proba = model.predict_proba(input_df)

# 6. Display Results
st.subheader('Prediction Result')
burnout_labels = np.array(['Low', 'Medium', 'High'])
result = burnout_labels[prediction][0]

if result == 'Low':
    st.success(f"Result: {result} Burnout Risk")
elif result == 'Medium':
    st.warning(f"Result: {result} Burnout Risk")
else:
    st.error(f"Result: {result} Burnout Risk")

# 7. Confidence Metrics
st.subheader('Prediction Confidence')
conf_df = pd.DataFrame(prediction_proba, columns=burnout_labels)
st.bar_chart(conf_df.T)