## config.py
from os import getenv

DB_URI = getenv('DATABASE_URL', 'sqlite:///./deribit_data.db')

DERIBIT_CLIENT_ID = getenv('DERIBIT_CLIENT_ID', 'B2uxTcMr')  # replace with your own or env var
DERIBIT_CLIENT_SECRET = getenv('DERIBIT_CLIENT_SECRET', '4DFWS6LcbQPBGU6xglFytGI0Bu8or0kRh-a8C-IjtGk')

MAX_CONCURRENT_REQUESTS = 10  # concurrency limit for async requests
