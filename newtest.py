import yfinance as yf
import numpy as np

start_year = 1994
end_year = 2023

tickers = ['AAPL', 'MSFT', 'GOOGL']  # Example tickers

#the list of years
years = list(range(start_year, end_year + 1))  # Create a list of years from 1994 to 2023
dsv_arrays1, dsv_arrays2, dsv_arrays3 = [], [], []

# the data is downloaded for each ticker and the average is calculated
yearly_data = []
for ticker in tickers:
    data = yf.download(ticker, start=f"{start_year}-01-01", end=f"{end_year}-12-31") 
    yearly_avg = data['Close'].resample('Y').mean()
    yearly_data.append(yearly_avg)

min_lenght =min(len(arr) for arr in yearly_data)


# each array is a list of yearly averages for each ticker
dsv_array1 = np.array([arr[:min_lenght] for arr in yearly_data])
dsv_array2 = np.array([arr[:min_lenght] for arr in yearly_data])
dsv_array3 = np.array([arr[:min_lenght] for arr in yearly_data])


print('years:', years)
print('APPL:', dsv_array1)
print('MSFT:', dsv_array2)
print('GOOGL:', dsv_array3)
