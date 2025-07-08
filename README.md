# Transformer-based Semantic Communication System

## Overview
This Proof of Concept implements a semantic communication system using BERT transformers, focusing on meaning preservation through noisy channels. The system compresses text into semantic embeddings for transmission and reconstructs it using neural decoders.

## Key Features
- BERT-based semantic encoding/decoding
- Channel noise simulation with configurable SNR
- Embedding compression/decompression networks
- End-to-end text reconstruction
- Semantic similarity preservation

## Implementation Details

### Semantic Encoder Architecture
```python
class SemanticEncoder(nn.Module):
    def __init__(self, bert_model_name='bert-base-uncased', hidden_dim=128):
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.compressor = nn.Sequential(
            nn.Linear(768, 256), 
            nn.ReLU(),
            nn.Linear(256, hidden_dim)
        )
```

### Semantic Decoder Architecture
```python
class SemanticDecoder(nn.Module):
    def __init__(self, bert_model_name='bert-base-uncased', hidden_dim=128):
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.decompressor = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 768)
        )
        self.output_layer = nn.Linear(768, self.bert.config.vocab_size)
```

### Channel Noise Simulation
```python
def add_channel_noise(embedding, snr_db=20):
    signal_power = torch.mean(embedding**2)
    noise_power = signal_power / (10**(snr_db/10))
    return embedding + torch.randn_like(embedding) * torch.sqrt(noise_power)
```

## Usage
1. Install dependencies:
```bash
pip install torch transformers numpy
```

2. Run the semantic communication test:
```bash
python semantic_comms.py
```

### Example Output
```
Original text: 5G networks require low latency and high reliability for critical applications.
Decoded text: 5g networks need low latency and high reliability for critical applications
```

## Mathematical Foundation
Signal-to-Noise Ratio (SNR) calculation:
```
SNR(dB) = 10 log10(P_signal / P_noise)
```

Reconstruction loss (cross-entropy):
```
L = -Σ y_true log(y_pred)
```

## References
1. [BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805)
2. [Semantic Communications: Principles and Challenges](https://arxiv.org/abs/2201.01389)
3. [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
4. [HuggingFace Transformers](https://huggingface.co/docs/transformers/index)

## System Architecture
```mermaid
graph LR
    A[Source Text] --> B[BERT Encoder]
    B --> C[Compression Network]
    C --> D[128-dim Embedding]
    D --> E[Channel Noise]
    E --> F[Decompression Network]
    F --> G[BERT Decoder]
    G --> H[Reconstructed Text]
    
    subgraph Transmitter
    A --> B --> C
    end
    
    subgraph Receiver
    F --> G --> H
    end
    
    style D fill:#f96
    style E fill:#699
```
