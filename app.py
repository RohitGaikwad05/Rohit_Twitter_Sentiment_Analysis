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

# Set Streamlit config
st.set_page_config(page_title="Twitter Sentiment Analyzer", layout="wide")

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
    url = "https://raw.githubusercontent.com/RohitGaikwad05/Rohitml/refs/heads/main/balanced_tweets_all_features.csv"
    column_names = ['target', 'id', 'date', 'query', 'user', 'text']
    df = pd.read_csv(url, encoding='latin-1', names=column_names)
    df['clean_text'] = df['text'].apply(clean_text)
    df['target'] = df['target'].apply(lambda x: 1 if x == 4 else 0)
    return df

df = load_data()

# ------------------ VECTORIZER & MODEL ------------------ #
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['clean_text'])
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=2000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# ------------------ SIDEBAR NAVIGATION ------------------ #
st.sidebar.title("📌 Navigation")
page = st.sidebar.radio("Go to", ["🏠 Overview", "📊 Data & Visualization", "🔮 Live Prediction"])

# ------------------ HEADER ------------------ #
st.markdown("<h1 style='text-align: center; color: #1DA1F2;'>💬 Twitter Sentiment Analyzer</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Analyze sentiments from tweets using Machine Learning and NLP</p>", unsafe_allow_html=True)
st.markdown("---")

# ------------------ OVERVIEW ------------------ #
if page == "🏠 Overview":
    st.subheader("🧠 Sentiment Classification using Logistic Regression")
    st.success(f"✅ **Model Accuracy:** {accuracy:.4f}")

    with st.expander("📋 Classification Report"):
        report = classification_report(y_test, y_pred, output_dict=True)
        st.dataframe(pd.DataFrame(report).transpose(), use_container_width=True)

    st.markdown("### 📌 Confusion Matrix")
    fig2, ax2 = plt.subplots()
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'],
                ax=ax2)
    ax2.set_xlabel("Predicted")
    ax2.set_ylabel("Actual")
    st.pyplot(fig2)

# ------------------ DATA AND VISUALIZATION ------------------ #
elif page == "📊 Data & Visualization":
    st.subheader("🔍 Preview Sample Data")
    st.dataframe(df[['target', 'text', 'clean_text']].sample(5), use_container_width=True)

    st.markdown("### 📊 Sentiment Distribution")
    col1, col2 = st.columns([2, 5])

    with col1:
        pos_count = df['target'].sum()
        neg_count = len(df) - pos_count
        st.metric("👍 Positive Tweets", f"{pos_count:,}")
        st.metric("👎 Negative Tweets", f"{neg_count:,}")

    with col2:
        fig, ax = plt.subplots()
        sns.countplot(x='target', data=df, palette='coolwarm', ax=ax)
        ax.set_xticklabels(['Negative', 'Positive'])
        ax.set_ylabel("Tweet Count")
        ax.set_xlabel("Sentiment")
        st.pyplot(fig)

# ------------------ LIVE PREDICTION ------------------ #
elif page == "🔮 Live Prediction":
    st.subheader("🔮 Predict Sentiment from Your Own Tweet")

    user_input = st.text_area("✍️ Enter your tweet text here:")

    if st.button("Predict Sentiment"):
        if user_input.strip() == "":
            st.warning("⚠️ Please enter a tweet to analyze.")
        else:
            cleaned_input = clean_text(user_input)
            vectorized_input = vectorizer.transform([cleaned_input])
            prediction = model.predict(vectorized_input)[0]
            label = "👍 Positive" if prediction == 1 else "👎 Negative"
            st.markdown(f"### 🎯 Predicted Sentiment: **{label}**")

# ------------------ FOOTER ------------------ #
st.markdown("---")
st.markdown("<p style='text-align:center;'>🚀 Built with ❤️ using Streamlit</p>", unsafe_allow_html=True)
