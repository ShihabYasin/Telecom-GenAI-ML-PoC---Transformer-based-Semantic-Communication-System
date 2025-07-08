import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
import numpy as np

class SemanticEncoder(nn.Module):
    def __init__(self, bert_model_name='bert-base-uncased', hidden_dim=128):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.compressor = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Linear(256, hidden_dim)
        )
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state
        cls_embedding = last_hidden_state[:, 0, :]  # [CLS] token embedding
        compressed = self.compressor(cls_embedding)
        return compressed

class SemanticDecoder(nn.Module):
    def __init__(self, bert_model_name='bert-base-uncased', hidden_dim=128):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.decompressor = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 768)
        )
        self.output_layer = nn.Linear(768, self.bert.config.vocab_size)
        
    def forward(self, compressed_embedding):
        decompressed = self.decompressor(compressed_embedding).unsqueeze(1)
        reconstructed = self.output_layer(decompressed)
        return reconstructed

def add_channel_noise(embedding, snr_db=20):
    """Add Gaussian noise to the embedding to simulate channel effects"""
    signal_power = torch.mean(embedding**2)
    noise_power = signal_power / (10**(snr_db/10))
    noise = torch.randn_like(embedding) * torch.sqrt(noise_power)
    return embedding + noise

def semantic_communication_test():
    # Initialize components
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    encoder = SemanticEncoder()
    decoder = SemanticDecoder()
    
    # Sample text
    text = "5G networks require low latency and high reliability for critical applications."
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    
    # Encode to semantic representation
    with torch.no_grad():
        compressed = encoder(inputs['input_ids'], inputs['attention_mask'])
    
    # Simulate channel transmission
    noisy_compressed = add_channel_noise(compressed, snr_db=15)
    
    # Decode back to text
    with torch.no_grad():
        logits = decoder(noisy_compressed)
        predictions = torch.argmax(logits, dim=-1)
        decoded_text = tokenizer.decode(predictions[0], skip_special_tokens=True)
    
    print("Original text:", text)
    print("Decoded text:", decoded_text)

if __name__ == "__main__":
    semantic_communication_test()
