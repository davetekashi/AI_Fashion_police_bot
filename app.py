from transformers import pipeline
from langchain import PromptTemplate, LLMChain, OpenAI, Cohere
import requests
import os
import streamlit as st



#load_dotenv(find_dotenv())
HUGGINGFACEHUB_API_TOKEN = "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"
#HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
COHERE_API_KEY = "Bgsr5bH8HujCbyM8LOkUR7UYjv1vBcGMAah9hc2f"

##Converting my image to text

def img2txt(url):
    image_to_text = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")

    text = image_to_text(url)[0]["generated_text"]

    print(text)
    return text


#img2txt("babe.JPG")


def generate_story(scenario):
    prompt_template = f"""
    You are a beauty and fashion expert that is very strict and also harsh;
    you must access the pictures thoroughly and explain in details how beautiful and ugly the person or people in the picture is and also give them a rating on a scale of 1-10 and the rating must be very strict;
    
    
    CONTEXT: {{scenario}}
    STORY:
    """

    prompt = PromptTemplate(template=prompt_template, input_variables=["scenario"])
    story_llm = LLMChain(llm=Cohere(temperature=0.2, verbose=True, cohere_api_key=COHERE_API_KEY), prompt=prompt, verbose=True)
    story = story_llm.predict(scenario=scenario)

    print(story)
    return story


#scenario = img2txt("try.jpg")
#story = generate_story(scenario)

def txt2speech(message):
    API_URL = "https://api-inference.huggingface.co/models/espnet/kan-bayashi_ljspeech_vits"
    headers = {"Authorization": f"Bearer {HUGGINGFACEHUB_API_TOKEN}"}
    payloads = {
        "inputs": message
   }

    response = requests.post(API_URL, headers = headers, json=payloads)
    with open('audio.flac', 'wb') as file:
        file.write(response.content)


#scenario = img2txt("try.JPG")
#story = generate_story(scenario)
#txt2speech(story)

def main():
    st.set_page_config(page_title="Dave AI Corp", page_icon="👾")

    st.header("Your AI beauty and fashion police")
    uploaded_file = st.file_uploader("choose image file......", type="jpg")

    if uploaded_file is not None:
        print(uploaded_file)
        bytes_data = uploaded_file.getvalue()
        with open(uploaded_file.name, "wb") as file:
            file.write(bytes_data)
        st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)
        scenario = img2txt(uploaded_file.name)
        story = generate_story(scenario)
        txt2speech(story)

        with st.expander("scenario"):
            st.write(scenario)
        with st.expander("story"):
            st.write(story)
        st.audio("audio.flac")

if __name__ == '__main__':
    main()

