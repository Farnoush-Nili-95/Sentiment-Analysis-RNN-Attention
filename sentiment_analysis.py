import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from datasets import load_dataset
from collections import Counter
import time
import sys

# --- Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
LEARNING_RATE = 0.001
BATCH_SIZE = 64
HIDDEN_DIM = 128
EMBEDDING_DIM = 100
NUM_LAYERS = 2
DROPOUT = 0.5
EPOCHS = 5
MAX_VOCAB_SIZE = 25000

# --- 1. Data Loading and Preprocessing ---

def load_data():
    print("Loading IMDB dataset...")
    try:
        dataset = load_dataset("imdb")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
    return dataset

def simple_tokenizer(text):
    return text.lower().split()

def build_vocab(dataset):
    print("Building vocabulary...")
    vocab_counter = Counter()
    for text in dataset['train']['text']:
        vocab_counter.update(simple_tokenizer(text))

    vocab = {"<pad>": 0, "<unk>": 1}
    for word, _ in vocab_counter.most_common(MAX_VOCAB_SIZE - 2):
        vocab[word] = len(vocab)
    return vocab

# Global Vocabulary
dataset = load_data()
vocab = build_vocab(dataset)
PAD_IDX = vocab['<pad>']
UNK_IDX = vocab['<unk>']

def text_pipeline(text):
    return [vocab.get(token, UNK_IDX) for token in simple_tokenizer(text)]

def label_pipeline(label):
    return float(label)

def collate_batch(batch):
    label_list, text_list = [], []
    for item in batch:
        label_list.append(label_pipeline(item['label']))
        processed_text = torch.tensor(text_pipeline(item['text']), dtype=torch.int64)
        text_list.append(processed_text)
    
    text_list_padded = pad_sequence(text_list, batch_first=True, padding_value=PAD_IDX)
    return (torch.tensor(label_list, dtype=torch.float32).to(DEVICE),
            text_list_padded.to(DEVICE),
            torch.tensor([len(t) for t in text_list], dtype=torch.int64).to(DEVICE))

# --- 2. Model Definitions ---

class BaselineLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, dropout, pad_idx):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers, 
                            bidirectional=True, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text, lengths):
        embedded = self.dropout(self.embedding(text))
        _, (hidden, cell) = self.lstm(embedded)
        hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1))
        return self.fc(hidden).squeeze(1)

class LSTMAttention(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, dropout, pad_idx):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers, 
                            bidirectional=True, dropout=dropout, batch_first=True)
        self.attention_layer = nn.Linear(hidden_dim * 2, 1)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text, lengths):
        embedded = self.dropout(self.embedding(text))
        lstm_output, (hidden, cell) = self.lstm(embedded)
        
        # Attention Mechanism
        attention_scores = self.attention_layer(lstm_output).squeeze(2)
        mask = (text != PAD_IDX).to(DEVICE)
        attention_scores = attention_scores.masked_fill(mask == 0, -1e10)
        attention_weights = F.softmax(attention_scores, dim=1)
        
        context_vector = torch.bmm(attention_weights.unsqueeze(1), lstm_output).squeeze(1)
        return self.fc(context_vector).squeeze(1)

# --- 3. Training Function ---

def train_and_evaluate(model):
    train_dataloader = DataLoader(dataset['train'], batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch)
    test_dataloader = DataLoader(dataset['test'], batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_batch)
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCEWithLogitsLoss().to(DEVICE)
    
    for epoch in range(EPOCHS):
        model.train()
        for batch in train_dataloader:
            labels, text, lengths = batch
            optimizer.zero_grad()
            predictions = model(text, lengths)
            loss = criterion(predictions, labels)
            loss.backward()
            optimizer.step()
            
    # Final Evaluation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in test_dataloader:
            labels, text, lengths = batch
            predictions = model(text, lengths)
            rounded_preds = torch.round(torch.sigmoid(predictions))
            correct += (rounded_preds == labels).sum().item()
            total += labels.size(0)
            
    return correct / total

if __name__ == "__main__":
    print(f"Training Baseline LSTM on {DEVICE}...")
    model1 = BaselineLSTM(len(vocab), EMBEDDING_DIM, HIDDEN_DIM, 1, NUM_LAYERS, DROPOUT, PAD_IDX).to(DEVICE)
    acc1 = train_and_evaluate(model1)
    print(f"Baseline Accuracy: {acc1*100:.2f}%")

    print(f"Training Attention LSTM on {DEVICE}...")
    model2 = LSTMAttention(len(vocab), EMBEDDING_DIM, HIDDEN_DIM, 1, NUM_LAYERS, DROPOUT, PAD_IDX).to(DEVICE)
    acc2 = train_and_evaluate(model2)
    print(f"Attention Accuracy: {acc2*100:.2f}%")