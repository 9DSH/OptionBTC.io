from Fetch_data import Fetching_data

db = Fetching_data()


    # Load public trades for that symbol in last 24h
trades_df = db.load_public_trades(symbol_filter=None, show_24h_public_trades=False)
print("Public Trades in last 24h for" , trades_df)
df = db.fetch_option_chain()
print(df)