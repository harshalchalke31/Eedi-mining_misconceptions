import re
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch

# Text cleaning function
def clean_text(text):
    text = re.sub(r'\\\\[a-zA-Z]+', '', str(text))  # Remove LaTeX commands
    text = re.sub(r'[^\w\s]', '', text)  # Remove special characters
    text = text.lower().strip()  # Lowercase and strip whitespace
    return text

# Generate embeddings using a transformer model
def generate_embeddings(text_list, model_name='bert-base-uncased'):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    
    # Tokenize and generate embeddings
    inputs = tokenizer(text_list, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    # Return CLS token as sentence embedding
    return outputs.last_hidden_state[:, 0, :].numpy()

# Encode categorical features
def encode_categorical(data, columns):
    encoded = pd.get_dummies(data[columns], drop_first=True)
    return encoded
