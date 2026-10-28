
import json
import pickle
import random
import numpy as np
import nltk
import matplotlib.pyplot as plt

from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping


# ---------------- CONFIGURATION ----------------

random.seed(42)
np.random.seed(42)

nltk.download("punkt")
nltk.download("wordnet")
nltk.download("punkt_tab")
nltk.download("omw-1.4")

lemmatizer = WordNetLemmatizer()
ignore_letters = {"?", "!", ".", ",", ";"}


# ---------------- LOAD DATASET ----------------

with open("intents.json", "r", encoding="utf-8") as file:
    intents = json.load(file)

words = []
classes = []
documents = []

for intent in intents["intents"]:
    for pattern in intent["patterns"]:
        tokens = nltk.word_tokenize(pattern)

        processed_words = [
            lemmatizer.lemmatize(word.lower())
            for word in tokens
            if word not in ignore_letters
        ]

        words.extend(processed_words)
        documents.append((processed_words, intent["tag"]))

    if intent["tag"] not in classes:
        classes.append(intent["tag"])

words = sorted(set(words))
classes = sorted(set(classes))


# ---------------- PREPARE DATA ----------------

X = []
y = []

for pattern_words, tag in documents:
    bag = [
        1 if word in pattern_words else 0
        for word in words
    ]

    X.append(bag)
    y.append(classes.index(tag))

X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.int32)

# One-hot encode output labels
y_one_hot = np.eye(len(classes), dtype=np.float32)[y]

# Split data into training and testing sets
X_train, X_test, y_train, y_test, labels_train, labels_test = (
    train_test_split(
        X,
        y_one_hot,
        y,
        test_size=0.40,
        random_state=42,
        stratify=y
    )
)
print("Total examples:", len(X))
print("Training examples:", len(X_train))
print("Testing examples:", len(X_test))
print("Vocabulary words:", len(words))
print("Intent classes:", len(classes))


# ---------------- SAVE VOCABULARY ----------------

with open("words.pkl", "wb") as file:
    pickle.dump(words, file)

with open("classes.pkl", "wb") as file:
    pickle.dump(classes, file)


# ---------------- BUILD MODEL ----------------

model = Sequential([
    Input(shape=(len(words),)),
    Dense(128, activation="relu"),
    Dropout(0.5),
    Dense(64, activation="relu"),
    Dropout(0.3),
    Dense(len(classes), activation="softmax")
])

model.compile(
    loss="categorical_crossentropy",
    optimizer=Adam(learning_rate=0.001),
    metrics=["accuracy"]
)


# ---------------- TRAIN MODEL ----------------

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=20,
    restore_best_weights=True
)

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    epochs=200,
    batch_size=8,
    callbacks=[early_stopping],
    verbose=1
)


# ---------------- EVALUATE MODEL ----------------

test_loss, test_accuracy = model.evaluate(
    X_test,
    y_test,
    verbose=0
)

print("\n----- MODEL EVALUATION -----")
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.2%}")


# Predict classes for test examples
predictions = model.predict(X_test, verbose=0)
predicted_labels = np.argmax(predictions, axis=1)

print("\n----- CLASSIFICATION REPORT -----")

print(
    classification_report(
        labels_test,
        predicted_labels,
        labels=np.arange(len(classes)),
        target_names=classes,
        zero_division=0
    )
)


# ---------------- CONFUSION MATRIX ----------------

cm = confusion_matrix(
    labels_test,
    predicted_labels,
    labels=np.arange(len(classes))
)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=classes
)

fig, ax = plt.subplots(figsize=(12, 10))

display.plot(
    ax=ax,
    xticks_rotation=90,
    cmap="Blues",
    colorbar=False
)

plt.title("Chatbot Intent Classification - Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=200)
plt.show()


# ---------------- ACCURACY GRAPH ----------------

plt.figure(figsize=(9, 5))

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.title("Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("accuracy_graph.png", dpi=200)
plt.close()
print("Accuracy graph saved successfully!")


# ---------------- LOSS GRAPH ----------------

plt.figure(figsize=(9, 5))

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.title("Model Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("loss_graph.png", dpi=200)
plt.close()
print("Loss graph saved successfully!")


# ---------------- SAVE MODEL ----------------

model.save("chatbot_model.keras")

print("\nChatbot model trained and saved successfully!")
print("Evaluation files saved:")
print("- accuracy_graph.png")
print("- loss_graph.png")
print("- confusion_matrix.png")

