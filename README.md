# AI Stock Predictor

An end-to-end web application that forecasts stock prices using a PyTorch LSTM (Long Short-Term Memory) deep learning model, built with Flask and styled with a clean minimalist interface.

---

## Features

* **Deep Learning Inference:** Utilizes a pre-trained PyTorch LSTM neural network to analyze historical stock trends and predict future prices.
* **Live Market Data:** Integrates with `yfinance` to fetch real-time historical data for any valid stock ticker.
* **Persistent Search History:** Automatically records past searches and predicted prices into a local SQLite database (`history.db`).
* **Visualized Analytics:** Generates dynamic, custom-styled matplotlib trend charts rendered directly on the frontend dashboard.
* **Universal Compatibility:** Configured with CPU device-mapping support so the model runs seamlessly across any environment or machine architecture.

---

## Project Structure

```text
AI-Stock-Predictor/
│
├── app.py                  # Main Flask application router and backend server
├── train.py                # Script used to train the LSTM model
├── prediction.py           # Standalone prediction helper script
├── lstm_stock_model.pth    # Pre-trained PyTorch model weights
├── requirements.txt        # Project dependencies
├── .gitignore              # Git ignore configuration
└── templates/
    └── index.html          # Frontend user interface
