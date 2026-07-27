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
    # Creates a file called history.db and builds the table if it doesn't exist
    conn = sqlite3.connect('history.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS search_history
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, 
                  ticker TEXT, 
                  predicted_price REAL)''')
    conn.commit()
    conn.close()

# Run this immediately when the app starts
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
model.load_state_dict(torch.load('lstm_stock_model.pth', weights_only=True))
model.eval()

# -----------------------------
# 3. Web Server Routes
# -----------------------------
@app.route('/')
def home():
    return render_template('index.html')

# --- Route to fetch the history ---
@app.route('/history', methods=['GET'])
def get_history():
    try:
        conn = sqlite3.connect('history.db')
        c = conn.cursor()
        # Grab the 5 most recent searches
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
        
        df = yf.download(ticker, period="6mo")
        if df.empty:
            return jsonify({'error': f"Could not find data for {ticker}"}), 400
            
        data = df['Close'].values.reshape(-1, 1)
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data)
        
        last_60_days = scaled_data[-60:]
        test_tensor = torch.tensor(last_60_days, dtype=torch.float32).unsqueeze(0).to(device)
        
        with torch.no_grad():
            prediction_scaled = model(test_tensor)
            
        predicted_price = scaler.inverse_transform(prediction_scaled.cpu().numpy())[0][0]
        real_prices = scaler.inverse_transform(last_60_days)

        # --- Save this prediction to the Database ---
        conn = sqlite3.connect('history.db')
        c = conn.cursor()
        c.execute("INSERT INTO search_history (ticker, predicted_price) VALUES (?, ?)", (ticker, float(predicted_price)))
        conn.commit()
        conn.close()

        # --- NEW: Generate the Chart Image with the minimalist aesthetic ---
        plt.figure(figsize=(8, 4))
        plt.plot(range(60), real_prices, color='#c91a25', label=f'{ticker} History (60 Days)')
        plt.scatter(60, predicted_price, color='white', s=100, zorder=5, label='AI Prediction')
        
        plt.title(f"{ticker} vs AI Prediction", color='white')
        plt.xlabel("Days", color='#aaa')
        plt.ylabel("Price (USD)", color='#aaa')
        plt.legend()
        plt.grid(color='#1a1a1a')
        
        ax = plt.gca()
        ax.set_facecolor('#050505')
        plt.gcf().patch.set_facecolor('#050505')
        ax.tick_params(colors='#aaa')

        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', bbox_inches='tight')
        img_buffer.seek(0)
        plt.close()
        
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')

        return jsonify({
            'price': float(predicted_price),
            'chart_image': img_base64
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("Starting Flask web server...")
    app.run(debug=True)