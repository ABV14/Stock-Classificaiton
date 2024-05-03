#This python file is used to run the predictions from the locally saved DistillbertSequenceClassifier


from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import pandas as pd 
# Load the model and tokenizer
model = AutoModelForSequenceClassification.from_pretrained("C:/Users/birru/Desktop/dtsc research/Project/Code/Execution Code/Final Code/Code/distillBert-pretrained/distilbert_finetuned_stock_sentiment")
tokenizer = AutoTokenizer.from_pretrained("C:/Users/birru/Desktop/dtsc research/Project/Code/Execution Code/Final Code/Code/distillBert-pretrained/distilbert_finetuned_stock_sentiment")
# Example dataset
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Choose device based on availability
dataset = pd.read_csv("C:/Users/birru/Desktop/dtsc research/Project/Code/Execution Code/Final Code/Data/Text Data/AMZN.csv")
label2id  = {
    "Bearish":0,
    "Bullish":1,
    "Neutral":2
}

id2label ={
    0: "Bearish",
    1: "Bullish",
    2: "Neutral"
}
def predict_sentiment(df, tokenizer, model):
    # Preprocess the text data
    texts = df['text'].tolist()
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

    # Move inputs to the appropriate device
    inputs = {key: tensor.to(device) for key, tensor in inputs.items()}

    # Predict sentiment labels
    with torch.no_grad():
        outputs = model(**inputs)

    # Map predicted label IDs to sentiment labels
    predicted_labels = torch.argmax(outputs.logits, dim=1).cpu().numpy()
    predicted_sentiments = [id2label[label_id] for label_id in predicted_labels]

    # Add predicted sentiment labels to the DataFrame
    df['predicted_sentiment'] = predicted_sentiments

    return df
predicted_df_amzn = predict_sentiment(dataset, tokenizer, model)
predicted_df_amzn.to_csv('Amazon predictions.csv', index=False) #Path to save your predictions