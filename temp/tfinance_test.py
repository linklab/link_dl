import yfinance as yf

# 주식 심볼을 지정합니다.
symbol = "AAPL"  # 예시로 애플 주식을 사용했습니다.

# Yahoo Finance에서 주식 정보를 가져옵니다.
stock = yf.Ticker(symbol)

# "quote volume"을 가져옵니다.
quote_volume = stock.history(period="1d")["Volume"].values[0]

# 결과를 출력합니다.
print(f"{symbol}의 현재 거래량: {quote_volume}")