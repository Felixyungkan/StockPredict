from flask import Flask, jsonify, render_template
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.linear_model import LinearRegression
import io
import base64

app = Flask(__name__)

def get_stock_data():
    tech_list = ['AMD']

    end = datetime.now()
    start = datetime(end.year - 1, end.month, end.day)

    stock_data = {}

    # Download data for each stock
    for stock in tech_list:
        stock_data[stock] = yf.download(stock, start, end)
        stock_data[stock]['company_name'] = stock  # Add company name column

    df = pd.concat(stock_data.values(), axis=0)


    df_reset = df.reset_index()
    df_reset.columns = [col[0] if col[1]=='' else f"{col[0]}_{col[1]}" for col in df_reset.columns]
    df_filter = df_reset.dropna(subset=['Date', 'Close_AMD'])
    df_filter['Days'] = (df_filter['Date'] - df_filter['Date'].min()).dt.days

    # Variabel independen (X) dan dependen (y)
    X = df_filter['Days'].values.reshape(-1, 1)
    y = df_filter['Close_AMD'].values

    # Model regresi linear
    LinearRegression_model = LinearRegression()
    LinearRegression_model.fit(X, y)

    # Prediksi harga berdasarkan tanggal
    y_pred = LinearRegression_model.predict(X)

    # Konversi kembali X ke format tanggal
    date_range = df_filter['Date'].min() + pd.to_timedelta(X.flatten(), unit='D')

    # Visualisasi
    plt.figure(figsize=(12, 6))
    plt.scatter(df_filter['Date'], df_filter['Close_AMD'], color='blue', label="Actual Data", alpha=0.6)

    plt.plot(date_range, y_pred, color='red', label="Linear Regression", linewidth=2)
    plt.axhline(df_filter['Close_AMD'].mean(), color='green', linestyle='dashed', linewidth=2, label="Average Price")

    plt.xlabel("Date")
    plt.ylabel("Close Price AMD (USD)")
    plt.title("Linear Regression Model (Date vs Close Price)")
    plt.legend()
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    # Simpan gambar ke buffer

    import os
    img = io.BytesIO()
    output_dir = "static/images"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "AMD.png")
    
    plt.savefig(output_path, format='png')
    plt.savefig(img, format='png')
    plt.close()
    img.seek(0)
    img_base64 = base64.b64encode(img.getvalue()).decode()
    
    return img_base64

@app.route('/')
def home():
    image = get_stock_data()
    return render_template('index.html', image=image)

if __name__ == '__main__':
    app.run(debug=True)
