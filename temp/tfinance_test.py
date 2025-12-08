import yfinance as yf

btc_data = yf.download('BTC-KRW', start='2014-09-17', end='2025-11-25', interval='1d')

# 데이터 저장
btc_data.to_csv('BTC_KRW_raw.csv')


# # 주식 심볼을 지정합니다.
# symbol = "AAPL"  # 예시로 애플 주식을 사용했습니다.
#
# # Yahoo Finance에서 주식 정보를 가져옵니다.
# stock = yf.Ticker(symbol)
#
# # "quote volume"을 가져옵니다.
# quote_volume = stock.history(period="1d")["Volume"].values[0]
#
# # 결과를 출력합니다.
# print(f"{symbol}의 현재 거래량: {quote_volume}")