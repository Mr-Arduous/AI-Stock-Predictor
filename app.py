from flask import Flask, render_template, request, jsonify
import yfinance as yf
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
import sqlite3

app = Flask(__name__)

# -----------------------------
# 0. Database Setup
# -----------------------------
def init_db():
    conn = sqlite3.connect('history.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS search_history
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, 
                  ticker TEXT, 
                  predicted_price REAL)''')
    conn.commit()
    conn.close()

init_db()

# -----------------------------
# 1. Rebuild the Network Architecture
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

# -----------------------------
# 2. Pre-load the AI Brain
# -----------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = StockLSTM().to(device)
model.load_state_dict(torch.load('lstm_stock_model.pth', map_location=torch.device('cpu'), weights_only=True))
model.eval()

# -----------------------------
# 3. Web Server Routes
# -----------------------------
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/history', methods=['GET'])
def get_history():
    try:
        conn = sqlite3.connect('history.db')
        c = conn.cursor()
        c.execute("SELECT ticker, predicted_price FROM search_history ORDER BY id DESC LIMIT 5")
        rows = c.fetchall()
        conn.close()
        
        history_list = [{'ticker': row[0], 'price': row[1]} for row in rows]
        return jsonify(history_list)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict_stock():
    try:
        data_in = request.get_json()
        ticker = data_in.get('ticker', 'AAPL').upper() if data_in else 'AAPL'
        horizon = int(data_in.get('horizon', 1))
        
        df = yf.download(ticker, period="1y")
        if df.empty:
            return jsonify({'error': f"Could not find data for {ticker}"}), 400
            
        data = df['Close'].values.reshape(-1, 1)
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data)
        
        last_60_days = scaled_data[-60:]
        
        # Multi-step forecasting loop
        predictions = []
        current_input = torch.tensor(last_60_days, dtype=torch.float32).unsqueeze(0).to(device)
        
        with torch.no_grad():
            for _ in range(horizon):
                pred_scaled = model(current_input)
                pred_value = pred_scaled.cpu().numpy()[0][0]
                predictions.append(pred_value)
                
                new_row = torch.tensor([[[pred_value]]], dtype=torch.float32).to(device)
                current_input = torch.cat((current_input[:, 1:, :], new_row), dim=1)
        
        predictions_real = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
        final_predicted_price = float(predictions_real[-1][0])
        real_prices = scaler.inverse_transform(last_60_days).flatten().tolist()
        flat_preds = [float(p[0]) for p in predictions_real]

        # Save to SQLite Database
        conn = sqlite3.connect('history.db')
        c = conn.cursor()
        c.execute("INSERT INTO search_history (ticker, predicted_price) VALUES (?, ?)", (ticker, final_predicted_price))
        conn.commit()
        conn.close()

        # Return raw data arrays for Chart.js interactive rendering
        return jsonify({
            'price': final_predicted_price,
            'history_prices': real_prices,
            'predictions': flat_preds,
            'horizon': horizon
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("Starting Flask web server...")
    app.run(debug=True)