import streamlit as st
import pickle

with open("emotion_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)


with open("label_map.pkl", "rb") as f:
    label_map = pickle.load(f)


st.set_page_config(page_title="Emotion Detection", page_icon="🧠")
st.title("🧠 Emotion Detection App")
st.write("Enter a sentence below, and the model will predict the underlying emotion:")


user_input = st.text_area("Enter your text here:")


if st.button("Predict Emotion"):
    if not user_input.strip():
        st.warning("Please enter some text before prediction.")
    else:
        
        input_vector = vectorizer.transform([user_input])

        
        pred = model.predict(input_vector)[0]

        
        emotion = label_map.get(pred, "Unknown")

        
        st.success(f"Predicted Emotion: **{emotion}** ")
