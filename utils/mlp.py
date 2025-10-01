import os

# For Huggingface
os.environ["TRANSFORMERS_CACHE"] = "/app/cache"
os.makedirs("/app/cache", exist_ok=True)

import torch
import joblib
from sentence_transformers import SentenceTransformer

class MLP(torch.nn.Module):
    def __init__(self, input_dim=3073, dropout_rate=0.5):
        super().__init__()
        self.fc1 = torch.nn.Linear(input_dim, 256)
        self.bn1 = torch.nn.BatchNorm1d(256)
        self.relu1 = torch.nn.ReLU()
        self.drop1 = torch.nn.Dropout(dropout_rate)
        self.fc2 = torch.nn.Linear(256, 64)
        self.bn2 = torch.nn.BatchNorm1d(64)
        self.relu2 = torch.nn.ReLU()
        self.drop2 = torch.nn.Dropout(dropout_rate * 0.7)
        self.fc_out = torch.nn.Linear(64, 1)

    def forward(self, x):
        x = self.drop1(self.relu1(self.bn1(self.fc1(x))))
        x = self.drop2(self.relu2(self.bn2(self.fc2(x))))
        return self.fc_out(x)

def load_model_and_metadata(model_path: str, model_type: str = "pytorch"):
    if model_type == "pytorch":
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        model = MLP(input_dim=3073)
        model.load_state_dict(checkpoint['model_state_dict'])
        best_threshold = checkpoint['best_threshold']
        label_encoder = checkpoint['label_encoder']
    else:  # sklearn
        checkpoint = joblib.load(model_path)
        model = checkpoint['model']
        best_threshold = checkpoint.get('best_threshold', 0.5)
        label_encoder = checkpoint['label_encoder']

    embedding_model = SentenceTransformer('all-mpnet-base-v2')
    return model, embedding_model, best_threshold, label_encoder
