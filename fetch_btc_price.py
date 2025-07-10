from deribit_client import DeribitClient
from config import DERIBIT_CLIENT_ID, DERIBIT_CLIENT_SECRET

client = DeribitClient(DERIBIT_CLIENT_ID, DERIBIT_CLIENT_SECRET)

def get_btcusd_price():
    price = client.fetch_btc_to_usd()
    highest_price , lowest_price = client.fetch_today_high_low()
    return price , highest_price, lowest_price