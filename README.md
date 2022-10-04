# Turkish-Sentiment-Analyser

This project takes the [distilled Turkish BERT](https://huggingface.co/dbmdz/distilbert-base-turkish-cased) model and fine-tunes it on the [sepidmnorozy/Turkish_sentiment](https://huggingface.co/datasets/sepidmnorozy/Turkish_sentiment) dataset for doing sentiment analysis. Additionally, for other people to use this model without code, Hugging Face Spaces and Streamlit are used to make an interactive web application.

## Performance of the model:
The model achieves the following results on the evaluation set:
* **Accuracy: 0.855**
* Loss: 0.4141
* F1: 0.8797

## Data:
Data for this project was obtained through 🤗/Datasets library. 

### Data Description:
* Dataset consists of three sets of "train", "test", and "validation".
	*  train: 4486 examples / number of rows
	*  test:  211 examples / number of rows
	*  validation: 105 examples / number of rows
* Each of the examples are labeled either as "LABEL_1" or "LABEL_0", corresponding to "positive" and "negative", respectively.
* Columns: 
	* label: whether a review is positive or negative
	* text: the text of the review

During the fine-tuning, due to limited storage, only random 2000 examples of the "train" and 200 examples of the "test" sets were used (all 105 examples of the "validation" set were used during evaluation).

## Web Application:
To make the web app, Streamlit is used to turn the Python script into an app. Then, for the final deployment, Hugging Face Spaces is used.

### Screenshot of the web page:
![Screenshot of the web page](https://github.com/azizbarank/Turkish-Sentiment-Analyser/blob/main/web_app.png)

## Usage

To use the model from the 🤗/transformers library

```python
# !pip install transformers

from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("azizbarank/distilbert-base-turkish-cased-sentiment")

model = AutoModelForSequenceClassification.from_pretrained("azizbarank/distilbert-base-turkish-cased-sentiment")
```
## Notes: 
* Google Colab is the environment used while obtaining the model.
* The link to the app: https://huggingface.co/spaces/azizbarank/Turkish-Sentiment-Analysis
* The link to the model: https://huggingface.co/azizbarank/distilbert-base-turkish-cased-sentiment
