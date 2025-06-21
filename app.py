# app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ------------------ CONFIG ------------------ #
st.set_page_config(page_title="Twitter(X) Sentiment Analyzer", layout="wide")
st.markdown("<h1 style='text-align: center; color: #1DA1F2;'>💬 Twitter Sentiment Analyzer</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Analyze Twitter sentiments using Logistic Regression & NLP</p>", unsafe_allow_html=True)
st.markdown("---")

# ------------------ SIDEBAR NAVIGATION ------------------ #
st.sidebar.title("📌 Menu")
page = st.sidebar.radio("Go to", ["🏠 Home", "📊 Data & Visualization", "🔮 Live Prediction"])

# ------------------ CLEANING FUNCTION ------------------ #
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r'\@\w+|\#', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ------------------ LOAD DATA ------------------ #
@st.cache_data
def load_data():
    try:
        url = "https://raw.githubusercontent.com/RohitGaikwad05/Rohitml/main/balanced_tweets_all_features.csv"
        column_names = ['target', 'id', 'date', 'query', 'user', 'text']
        df = pd.read_csv(url, encoding='latin-1', names=column_names, dtype={'target': str, 'id': str}, low_memory=False)
        df['clean_text'] = df['text'].apply(clean_text)
        df['target'] = df['target'].apply(lambda x: 1 if x == '4' else 0)
        return df
    except Exception as e:
        st.error(f"❌ Failed to load data: {e}")
        st.stop()

df = load_data()

# ------------------ MODEL TRAINING ------------------ #
@st.cache_resource
def train_model(data):
    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(data['clean_text'])
    y = data['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegression(max_iter=2000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return model, vectorizer, X_test, y_test, y_pred, accuracy

model, vectorizer, X_test, y_test, y_pred, accuracy = train_model(df)

# ------------------ PAGE: HOME ------------------ #
if page == "🏠 Home":
    st.title("🧠 Twitter(X) Sentiment Analysis with Logistic Regression")
    st.markdown("""
    This web app uses a machine learning model (Logistic Regression) to analyze Twitter(X) sentiments.

    **Features**:
    - Cleans and preprocesses raw tweet text
    - Transforms text using TF-IDF vectorization
    - Trains on labeled dataset to classify sentiment
    - Displays model performance and predictions interactively

    **Labels**:
    - `0`: Negative 😞  
    - `1`: Positive 😊

    **Built with**:
    - Scikit-learn
    - Streamlit
    - Matplotlib & Seaborn
    """)

# ------------------ PAGE: DATA & VISUALIZATION ------------------ #
elif page == "📊 Data & Visualization":
    st.subheader("🔍 Preview Sample Data")
    st.dataframe(df[['target', 'text', 'clean_text']].sample(5), use_container_width=True)

    st.subheader("🧠 Model Performance")
    st.success(f"✅ **Model Accuracy:** {accuracy:.4f}")

    st.markdown("### 📋 Classification Report")
    with st.expander("View Full Report"):
        report = classification_report(y_test, y_pred, output_dict=True)
        st.dataframe(pd.DataFrame(report).transpose(), use_container_width=True)

    st.markdown("### 📌 Confusion Matrix")
    fig, ax = plt.subplots()
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

    st.markdown("### 📊 Sentiment Distribution")
    col1, col2 = st.columns([2, 5])

    with col1:
        pos_count = df['target'].sum()
        neg_count = len(df) - pos_count
        st.metric("👍 Positive Tweets", f"{pos_count:,}")
        st.metric("👎 Negative Tweets", f"{neg_count:,}")

    with col2:
        fig, ax = plt.subplots()
        sns.countplot(x='target', data=df, hue='target', palette='coolwarm', ax=ax, legend=False)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Negative', 'Positive'])
        ax.set_ylabel("Tweet Count")
        ax.set_xlabel("Sentiment")
        st.pyplot(fig)

# ------------------ PAGE: LIVE PREDICTION ------------------ #
elif page == "🔮 Live Prediction":
    st.subheader("🔮 Predict the Sentiment of Your Tweet")
    tweet = st.text_area("✍️ Enter your tweet:")

    if st.button("Predict"):
        if tweet.strip() == "":
            st.warning("⚠️ Please enter a tweet.")
        else:
            cleaned = clean_text(tweet)
            vec_input = vectorizer.transform([cleaned])
            prediction = model.predict(vec_input)[0]
            emoji = "👍" if prediction == 1 else "👎"
            sentiment = "Positive" if prediction == 1 else "Negative"
            st.success(f"🎯 **Predicted Sentiment:** {emoji} {sentiment}")

# ------------------ FOOTER ------------------ #
st.markdown("---")
st.markdown("<p style='text-align:center;'>🚀 Built with ❤️ using Streamlit | Logistic Regression + NLP</p>", unsafe_allow_html=True)
