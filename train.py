import yfinance as yf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import torch.optim as optim

# -----------------------------
# 1. Fetch & Prepare Stock Data
# -----------------------------
ticker = "AAPL"
print(f"Downloading stock data for {ticker}...")

# Fetch 4 years of Apple stock data
df = yf.download(ticker, start='2020-01-01', end='2024-01-01')

# Extract the 'Close' column safely
data = df['Close'].values.reshape(-1, 1)

# Scale prices between 0 and 1
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# Create 60-day historical sequences
sequence_length = 60
X, y = [], []

for i in range(sequence_length, len(scaled_data)):
    X.append(scaled_data[i-sequence_length:i, 0])
    y.append(scaled_data[i, 0])

X = np.reshape(np.array(X), (len(X), sequence_length, 1))
y = np.array(y)

X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

# -----------------------------
# 2. Define the PyTorch Neural Network
# -----------------------------
class StockLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=2, output_size=1):
        super(StockLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# Target your RTX 3050 GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Running training on: {device}")

model = StockLSTM().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Move data to GPU VRAM
X_gpu = X_tensor.to(device)
y_gpu = y_tensor.to(device)

# -----------------------------
# 3. Train the Model (100 Epochs)
# -----------------------------
print("Training AI model...")
for epoch in range(100):
    model.train()
    optimizer.zero_grad()
    
    outputs = model(X_gpu)
    loss = criterion(outputs, y_gpu)
    
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch + 1}/100], Loss: {loss.item():.6f}")

# Save the trained model file
torch.save(model.state_dict(), 'lstm_stock_model.pth')
print("Finished! Model saved as 'lstm_stock_model.pth'.")