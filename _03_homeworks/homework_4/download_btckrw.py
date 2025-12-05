import yfinance as yf

# Bitcoin 데이터 다운로드
btc_data = yf.download('BTC-KRW', start='2014-09-17', end='2025-12-01', interval='1d')

# 데이터 저장
btc_data.to_csv('BTC_KRW.csv')
