import os
os.system("pip install torch")
os.system("pip install transformers")
os.system("pip install sentencepiece")

import streamlit as st
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("azizbarank/distilbert-base-turkish-cased-sentiment")
model = AutoModelForSequenceClassification.from_pretrained("azizbarank/distilbert-base-turkish-cased-sentiment")

def classify(text):
    cls= pipeline("text-classification",model=model, tokenizer=tokenizer)
    return cls(text)[0]['label']
    

site_header = st.container()
text_input = st.container()
model_results = st.container()

with site_header:
    st.title('Turkish Sentiment Analysis 😀😠')
    st.markdown(
    """
    [Distilled Turkish BERT model](https://huggingface.co/dbmdz/distilbert-base-turkish-cased) that I fine-tuned on the [sepidmnorozy/Turkish_sentiment](https://huggingface.co/datasets/sepidmnorozy/Turkish_sentiment) dataset that is heavily based on different reviews about services/places.
    
    For more information on the dataset:
    
    * [Hugging Face](https://huggingface.co/datasets/sepidmnorozy/Turkish_sentiment)
    """
)

with text_input:
    st.header('Is Your Review Considered Positive or Negative?')
    st.write("""*Please note that predictions are based on how the model was trained, so it may not be an accurate representation.*""")
    user_text = st.text_input('Enter Text', max_chars=300)

with model_results:    
    st.subheader('Prediction:')
    if user_text:
        prediction = classify(user_text)
    
        if prediction == "LABEL_0":
            st.subheader('**Negative**')
        else:
            st.subheader('**Positive**')
        st.text('')
