
import streamlit as st
import dotenv
import os
from PIL import Image
from audio_recorder_streamlit import audio_recorder
import base64
from io import BytesIO
import google.generativeai as genai
import random

from google.generativeai.types import HarmCategory, HarmBlockThreshold
from dotenv import load_dotenv
load_dotenv()

# Define safety settings to allow for less restrictive filtering
safety_settings = {
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    # HarmCategory.HARM_CATEGORY_VIOLENCE: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
}

google_models = [
    "gemini-1.5-flash",
    "gemini-1.5-pro",
]

# Function to convert the messages format from Streamlit to Gemini
def messages_to_gemini(messages):
    gemini_messages = []
    prev_role = None
    for message in messages:
        if prev_role and (prev_role == message["role"]):
            gemini_message = gemini_messages[-1]
        else:
            gemini_message = {
                "role": "model" if message["role"] == "assistant" else "user",
                "parts": [],
            }

        for content in message["content"]:
            if content["type"] == "text":
                gemini_message["parts"].append(content["text"])
            elif content["type"] == "image_url":
                gemini_message["parts"].append(base64_to_image(content["image_url"]["url"]))
            elif content["type"] == "video_file":
                gemini_message["parts"].append(genai.upload_file(content["video_file"]))
            elif content["type"] == "audio_file":
                gemini_message["parts"].append(genai.upload_file(content["audio_file"]))

        if prev_role != message["role"]:
            gemini_messages.append(gemini_message)

        prev_role = message["role"]
        
    return gemini_messages

# Function to query and stream the response from the LLM with error handling for blocked responses
def stream_llm_response(model_params, api_key=None):
    response_message = ""
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(
        model_name=model_params["model"],
        generation_config={
            "temperature": model_params["temperature"] if "temperature" in model_params else 0.3,
            "safety_settings": safety_settings
        }
    )
    gemini_messages = messages_to_gemini(st.session_state.messages)

    try:
        for chunk in model.generate_content(
            contents=gemini_messages,
            stream=True,
        ):
            if chunk.safety_ratings:
                st.warning("⚠️ Response blocked due to safety filters. Try rephrasing your query.")
                break

            chunk_text = chunk.text or ""
            response_message += chunk_text
            yield chunk_text

    except ValueError as e:
        st.error("Response blocked due to safety settings. Please adjust the content or safety settings.")
        print(f"Error: {e}")

    st.session_state.messages.append({
        "role": "assistant", 
        "content": [
            {
                "type": "text",
                "text": response_message,
            }
        ]
    })

# Function to convert file to base64
def get_image_base64(image_raw):
    buffered = BytesIO()
    image_raw.save(buffered, format=image_raw.format)
    img_byte = buffered.getvalue()
    return base64.b64encode(img_byte).decode('utf-8')

def file_to_base64(file):
    with open(file, "rb") as f:
        return base64.b64encode(f.read())

def base64_to_image(base64_string):
    base64_string = base64_string.split(",")[1]
    return Image.open(BytesIO(base64.b64decode(base64_string)))

def main():
    st.set_page_config(
        page_title="GeminiChat",
        page_icon="💠",
        layout="centered",
        initial_sidebar_state="expanded",
    )

    st.html("""<h1 style="text-align: center; color: #6ca395;">🤖 <i>GeminiChat: An AI-Powered Chat </i></h1>""")

    google_api_key = os.getenv("GOOGLE_API_KEY")
    
    if google_api_key == "" or google_api_key is None:
        st.write("#")
        st.warning("⬅️ Please introduce an API Key to continue...")
    else:
        if "messages" not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                for content in message["content"]:
                    if content["type"] == "text":
                        st.write(content["text"])
                    elif content["type"] == "image_url":      
                        st.image(content["image_url"]["url"])
                    elif content["type"] == "video_file":
                        st.video(content["video_file"])
                    elif content["type"] == "audio_file":
                        st.audio(content["audio_file"])

        with st.sidebar:
            st.title("Menu:")
            st.sidebar.markdown("""<hr style="border:1px solid #16a085">""", unsafe_allow_html=True)
 
            available_models = google_models if google_api_key else []
            model = st.selectbox("Select a model:", available_models, index=0)
            
            with st.popover("⚙️ Model parameters"):
                model_temp = st.slider("Temperature", min_value=0.0, max_value=2.0, value=0.3, step=0.1)

            audio_response = st.toggle("Audio response", value=False)
            if audio_response:
                cols = st.columns(2)
                with cols[0]:
                    tts_voice = st.selectbox("Select a voice:", ["alloy", "echo", "fable", "onyx", "nova", "shimmer"])
                with cols[1]:
                    tts_model = st.selectbox("Select a model:", ["tts-1", "tts-1-hd"], index=1)

            model_params = {
                "model": model,
                "temperature": model_temp,
            }

            def reset_conversation():
                if "messages" in st.session_state and len(st.session_state.messages) > 0:
                    st.session_state.pop("messages", None)

            st.button(
                "🗑️ Reset conversation", 
                on_click=reset_conversation,
            )

            st.sidebar.markdown("""<hr style="border:1px solid #16a085">""", unsafe_allow_html=True)

            st.write(f"### **🖼️ Add an image or a video file:**")

            def add_image_to_messages():
                if st.session_state.uploaded_img or ("camera_img" in st.session_state and st.session_state.camera_img):
                    img_type = st.session_state.uploaded_img.type if st.session_state.uploaded_img else "image/jpeg"
                    if img_type == "video/mp4":
                        video_id = random.randint(100000, 999999)
                        with open(f"video_{video_id}.mp4", "wb") as f:
                            f.write(st.session_state.uploaded_img.read())
                        st.session_state.messages.append(
                            {
                                "role": "user", 
                                "content": [{
                                    "type": "video_file",
                                    "video_file": f"video_{video_id}.mp4",
                                }]
                            }
                        )
                    else:
                        raw_img = Image.open(st.session_state.uploaded_img or st.session_state.camera_img)
                        img = get_image_base64(raw_img)
                        st.session_state.messages.append(
                            {
                                "role": "user", 
                                "content": [{
                                    "type": "image_url",
                                    "image_url": {"url": f"data:{img_type};base64,{img}"}
                                }]
                            }
                        )

            cols_img = st.columns(2)

            with cols_img[0]:
                with st.popover("📁 Upload"):
                    st.file_uploader(
                        "Upload an image or a video:", 
                        type=["png", "jpg", "jpeg", "mp4"], 
                        accept_multiple_files=False,
                        key="uploaded_img",
                        on_change=add_image_to_messages,
                    )

            with cols_img[1]:                    
                with st.popover("📸 Camera"):
                    activate_camera = st.checkbox("Activate camera")
                    if activate_camera:
                        st.camera_input(
                            "Take a picture", 
                            key="camera_img",
                            on_change=add_image_to_messages,
                        )
            st.sidebar.markdown(""" 
                        <hr style="border:1px solid #16a085">
                        <footer style="text-align:center; color: #7f8c8d;">
                            <p>Powered by Google Gemini | © 2024 Sarvesh Udapurkar</p>
                        </footer>
                    """, unsafe_allow_html=True)
        

        if prompt := st.chat_input():
            st.session_state.messages.append({
                "role": "user",
                "content": [{
                    "type": "text",
                    "text": prompt,
                }]
            })

        if st.session_state.messages:
            with st.chat_message("assistant"):
                for content in stream_llm_response(model_params, google_api_key):
                    st.write(content)

if __name__ == "__main__":
    main()
