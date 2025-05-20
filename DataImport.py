import yfinance as yf
import matplotlib.pyplot as plt
#imports the libraries needed to pull the data and interpret it

def get_stock_data(ticker_symbol):
    # Download data based on the ticker symbol that was inputted
    stock = yf.Ticker(ticker_symbol)

    try:
        stock_info = stock.info
        # Get closing price data for the last 7 days
        hist = stock.history(period="7d")

        # Outputs the information for the stock
        print(f"\nCompany: {stock_info.get('longName', 'N/A')}")
        print(f"Symbol: {stock_info.get('symbol', 'N/A')}")
        print(f"Current Price: ${stock_info.get('currentPrice', 'N/A')}")
        print(f"Sector: {stock_info.get('sector', 'N/A')}")

        # chart closing prices
        plt.figure(figsize=(12, 6))
        plt.plot(hist.index, hist['Close'], label='Daily Close Price', color='blue')
        plt.title(f'{ticker_symbol} - Last 7 Days Closing Prices')
        plt.xlabel('Date and Time')
        plt.ylabel('Price (USD)')
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

    except Exception as e:
        print("Error retrieving data:", e)

# Asks for input then runs the commands to fetch the data based on the stock
if __name__ == "__main__":
    symbol = input("Enter stock symbol (e.g., AAPL, MSFT): ").upper()
    get_stock_data(symbol)