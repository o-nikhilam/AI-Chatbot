
import json
import pickle
import random

import numpy as np
import nltk
import streamlit as st

from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model


# ---------------- PAGE CONFIGURATION ----------------

st.set_page_config(
    page_title="AI Chatbot | Deep Learning",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ---------------- LOAD MODEL AND DATA ----------------

@st.cache_resource
def load_chatbot():
    model = load_model("chatbot_model.keras")

    with open("words.pkl", "rb") as file:
        words = pickle.load(file)

    with open("classes.pkl", "rb") as file:
        classes = pickle.load(file)

    with open("intents.json", "r", encoding="utf-8") as file:
        intents = json.load(file)

    return model, words, classes, intents


model, words, classes, intents = load_chatbot()
lemmatizer = WordNetLemmatizer()


# ---------------- CUSTOM STYLING ----------------

st.markdown(
    """
    <style>
    .stApp {
        background-color: #0e1117;
        color: #f5f7fb;
    }

    .main-title {
        font-size: 42px;
        font-weight: 800;
        text-align: center;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        color: #a8b3cf;
        font-size: 16px;
        margin-bottom: 25px;
    }

    .info-card {
        background: #1b2230;
        padding: 18px;
        border-radius: 14px;
        border: 1px solid #30394d;
        margin-bottom: 12px;
    }

    .stButton > button {
        border-radius: 10px;
        min-height: 42px;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ---------------- CHATBOT FUNCTIONS ----------------

def preprocess_text(sentence):
    tokens = nltk.word_tokenize(sentence)

    return [
        lemmatizer.lemmatize(word.lower())
        for word in tokens
        if word not in {"?", "!", ".", ",", ";"}
    ]


def predict_intent(sentence):
    tokens = preprocess_text(sentence)

    bag = [
        1 if word in tokens else 0
        for word in words
    ]

    prediction = model.predict(
        np.array([bag], dtype=np.float32),
        verbose=0
    )[0]

    best_index = int(np.argmax(prediction))
    confidence = float(prediction[best_index])

    if confidence < 0.50:
        return None, confidence

    return classes[best_index], confidence


def get_response(tag):
    for intent in intents["intents"]:
        if intent["tag"] == tag:
            return random.choice(intent["responses"])

    return "Sorry, I could not find a suitable answer."


# ---------------- SIDEBAR ----------------

with st.sidebar:
    st.title("🤖 AI Assistant")

    st.markdown("---")

    st.subheader("📚 About Project")
    st.write(
        "An intent-classification chatbot built using "
        "Deep Learning, Natural Language Processing, "
        "TensorFlow and Streamlit."
    )

    st.subheader("🛠️ Technologies")
    st.markdown(
        "- Python\n"
        "- TensorFlow / Keras\n"
        "- NLTK\n"
        "- Neural Networks\n"
        "- Streamlit"
    )

    st.markdown("---")

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": (
                    "Hello! 👋 I am your AI Assistant. "
                    "Ask me about AI, Python, Deep Learning or NLP."
                )
            }
        ]
        st.rerun()

    st.caption("Deep Learning College Project")


# ---------------- MAIN INTERFACE ----------------

st.markdown(
    '<div class="main-title">🤖 AI Chatbot</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Your intelligent learning assistant powered by Deep Learning'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="info-card">
        👋 Welcome! Ask questions about Artificial Intelligence,
        Machine Learning, Deep Learning, Python, NLP and Data Science.
    </div>
    """,
    unsafe_allow_html=True
)


# ---------------- SUGGESTED QUESTIONS ----------------

st.subheader("💡 Try asking")

suggestions = [
    "What is AI?",
    "What is Python?",
    "What is Deep Learning?",
    "What is CNN?"
]

columns = st.columns(4)

if "pending_question" not in st.session_state:
    st.session_state.pending_question = None

for index, question in enumerate(suggestions):
    if columns[index].button(question, use_container_width=True):
        st.session_state.pending_question = question


# ---------------- CHAT HISTORY ----------------

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Hello! 👋 I am your AI Assistant. "
                "How can I help you today?"
            )
        }
    ]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# ---------------- USER INPUT ----------------

user_input = st.chat_input("Type your message here...")

if st.session_state.pending_question:
    user_input = st.session_state.pending_question
    st.session_state.pending_question = None


if user_input:
    st.session_state.messages.append(
        {"role": "user", "content": user_input}
    )

    with st.chat_message("user"):
        st.markdown(user_input)

    tag, confidence = predict_intent(user_input)

    if tag is None:
        response = (
            "I'm not sure about that question yet. "
            "Please try asking about AI, Python, Machine Learning, "
            "Deep Learning, CNN, NLP or Data Science."
        )
    else:
        response = get_response(tag)

    st.session_state.messages.append(
        {"role": "assistant", "content": response}
    )

    with st.chat_message("assistant"):
        st.markdown(response)

        if tag is not None:
            st.caption(
                f"Predicted intent: {tag} | "
                f"Model confidence: {confidence:.1%}"
            )

