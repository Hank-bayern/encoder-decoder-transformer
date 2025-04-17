import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import csv, ast
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ---------------- Dataset ----------------
class SequenceClassificationDataset(Dataset):
    def __init__(self, csv_file):
        self.inputs, self.outputs = [], []
        with open(csv_file, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                input_str, output_str = row[0].split('][')
                self.inputs.append(ast.literal_eval(input_str + ']'))
                self.outputs.append(ast.literal_eval('[' + output_str))
        self.inputs = torch.tensor(self.inputs, dtype=torch.long)
        self.outputs = torch.tensor(self.outputs, dtype=torch.long)

    def __len__(self): return len(self.inputs)
    def __getitem__(self, idx): return self.inputs[idx], self.outputs[idx]

# ---------------- Model ----------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2], pe[:, 1::2] = torch.sin(position * div_term), torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
    def forward(self, x): return self.dropout(x + self.pe[:x.size(0), :])

class TransformerSeq2Seq(nn.Module):
    def __init__(self, vocab_size, d_model=256, nhead=4, num_encoder_layers=4, num_decoder_layers=4, dim_feedforward=1024, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        self.pos_decoder = PositionalEncoding(d_model, dropout)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=False)
        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=False)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        self.output_layer = nn.Linear(d_model, vocab_size)

    def forward(self, src, tgt):
        src, tgt = self.embedding(src).transpose(0, 1), self.embedding(tgt).transpose(0, 1)
        memory = self.encoder(self.pos_encoder(src))
        output = self.decoder(self.pos_decoder(tgt), memory)
        return self.output_layer(output).transpose(0, 1)

# ---------------- Inference  ----------------
def run_inference(test_path, model_path, output_csv, vocab_size):
    dataset = SequenceClassificationDataset(test_path)
    loader = DataLoader(dataset, batch_size=1)
    model = TransformerSeq2Seq(vocab_size).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    results = []
    with torch.no_grad():
        for inp, tgt in loader:
            inp, tgt = inp.to(device), tgt.to(device)
            decoder_input = tgt[:, :-1]
            pred = model(inp, decoder_input).argmax(dim=-1)
            pred = torch.cat([tgt[:, :1], pred], dim=1)  

            results.append({
                "input": inp[0].cpu().tolist(),
                "true_output": tgt[0].cpu().tolist(),
                "pred_output": pred[0].cpu().tolist()  
            })

    
    df = pd.DataFrame(results)
    df['pred_output'] = df['pred_output'].apply(lambda x: x if isinstance(x, list) else [x])
    df.to_csv(output_csv, index=False)
    print(f" test result has been saved to: {output_csv}")

# ---------------- Run ----------------
run_inference(
    test_path="E:/Language Model Project/final-model/translation_mistake_15-20.csv",
    model_path="E:\Language Model Project\\final-model\\result\\expanded 10 times\\20epochs\\best_model.pth",
    output_csv="E:\Language Model Project\\final-model\\result\\expanded 10 times\\20epochs\\translation_mistake_result.csv",
    vocab_size=134606  # vocab size training
)