import yfinance as yf
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler

# -----------------------------
# 1. Rebuild the Network Shell
# -----------------------------
# We have to define the exact same architecture so the saved weights have a place to go
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

# -----------------------------
# 2. Load the Saved Brain
# -----------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = StockLSTM().to(device)

# Load the weights from the file you just created
model.load_state_dict(torch.load('lstm_stock_model.pth', weights_only=True))
model.eval()  # Lock the model into "evaluation mode" so it stops learning and just predicts

# -----------------------------
# 3. Get Fresh Data
# -----------------------------
print("Fetching the most recent 60 days of Apple stock...")
df = yf.download("AAPL", period="6mo") # Get recent data
data = df['Close'].values.reshape(-1, 1)

# Scale the data exactly like we did in training
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# Grab ONLY the last 60 days
last_60_days = scaled_data[-60:]
# Shape it for the GPU: [Batch=1, Sequence=60, Features=1]
test_tensor = torch.tensor(last_60_days, dtype=torch.float32).unsqueeze(0).to(device)

# -----------------------------
# 4. Make the Prediction
# -----------------------------
with torch.no_grad(): # Tell PyTorch not to calculate gradients (saves VRAM)
    prediction_scaled = model(test_tensor)

# The AI spits out a decimal between 0 and 1. We need to convert it back to real dollars.
predicted_price = scaler.inverse_transform(prediction_scaled.cpu().numpy())

print("\n========================================")
print(f"  AI PREDICTED NEXT CLOSING PRICE: ${predicted_price[0][0]:.2f}")
print("========================================")
import matplotlib.pyplot as plt

# Inverse transform the last 60 days back to real dollars for the chart
real_prices = scaler.inverse_transform(last_60_days)

# Render the graph
plt.figure(figsize=(10, 5))
plt.plot(range(60), real_prices, color='blue', label='Actual Price (Last 60 Days)')
plt.scatter(60, predicted_price[0][0], color='red', s=100, label='AI Prediction (Day 61)')
plt.title("Apple Stock: 60-Day History vs AI Prediction")
plt.xlabel("Days")
plt.ylabel("Price (USD)")
plt.legend()
plt.grid(True)
plt.show()