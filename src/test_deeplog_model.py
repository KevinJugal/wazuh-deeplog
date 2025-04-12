import torch
import torch.nn as nn

# Match training setup
EMBEDDING_DIM = 64
HIDDEN_SIZE = 64

class DeepLog(nn.Module):
    def __init__(self, num_classes, embedding_dim, hidden_size):
        super(DeepLog, self).__init__()
        self.embedding = nn.Embedding(num_classes + 1, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes + 1)

    def forward(self, x):
        x = self.embedding(x)
        _, (h_n, _) = self.lstm(x)
        out = self.fc(h_n[-1])
        return out

if __name__ == "__main__":
    model_path = "deeplog_model_new.pt"
    checkpoint = torch.load(model_path, map_location=torch.device("cpu"))

    output_dim = checkpoint['fc.weight'].shape[0]
    num_classes = output_dim - 1

    model = DeepLog(num_classes, EMBEDDING_DIM, HIDDEN_SIZE)
    model.load_state_dict(checkpoint)
    model.eval()

    print("[+] Model loaded and ready to test.")
    print(model)
