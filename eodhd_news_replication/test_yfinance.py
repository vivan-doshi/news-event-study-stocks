import yfinance as yf
from curl_cffi import requests

session = requests.Session(impersonate="chrome")  # key trick

data = yf.download(
    "AAPL",
    start="2024-01-01",
    end="2024-02-01",
    interval="1d",
    session=session,
    progress=False,
)

print(data.head())
print("Empty?", data.empty)
