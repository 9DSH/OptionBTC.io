import asyncio
import logging
import requests
import aiohttp
from datetime import datetime, timedelta
from sqlalchemy.exc import IntegrityError
from concurrent.futures import ThreadPoolExecutor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import VotingClassifier
import xgboost as xgb
import lightgbm as lgb
from skopt import BayesSearchCV
from sklearn.utils import resample
import pandas as pd

from db import SessionLocal, OptionChain, PublicTrade
from config import DERIBIT_CLIENT_ID, DERIBIT_CLIENT_SECRET, MAX_CONCURRENT_REQUESTS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DeribitClient:
    def __init__(self, client_id, client_secret):
        self.client_id = client_id
        self.client_secret = client_secret
        self.access_token = None
        self.session = requests.Session()
        self.executor = ThreadPoolExecutor(max_workers=5)
        self.semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

    def authenticate(self):
        if self.access_token:
            return self.access_token
        url = 'https://www.deribit.com/api/v2/public/auth'
        params = {
            'client_id': self.client_id,
            'client_secret': self.client_secret,
            'grant_type': 'client_credentials'
        }
        try:
            resp = self.session.get(url, params=params)
            resp.raise_for_status()
            self.access_token = resp.json().get('result', {}).get('access_token')
            if not self.access_token:
                logger.error("Authentication failed: No access token received")
            return self.access_token
        except requests.RequestException as e:
            logger.error(f"Authentication error: {e}")
            return None

    async def fetch_option_instruments(self, currency='BTC'):
        token = self.authenticate()
        if not token:
            logger.error("Cannot fetch option instruments without valid access token")
            return []
        headers = {'Authorization': f'Bearer {token}'}
        url = 'https://www.deribit.com/api/v2/public/get_instruments'
        params = {'currency': currency, 'kind': 'option', 'expired': 'false'}
        async with aiohttp.ClientSession(headers=headers) as session:
            async with session.get(url, params=params) as resp:
                resp.raise_for_status()
                result = await resp.json()
                instruments = result.get('result', [])
                logger.info(f"Fetched {len(instruments)} option instruments.")
                return instruments

    async def fetch_order_book(self, session_http, instrument_name):
        url = 'https://www.deribit.com/api/v2/public/get_order_book'
        params = {'instrument_name': instrument_name}
        async with self.semaphore:
            for attempt in range(3):
                try:
                    async with session_http.get(url, params=params) as resp:
                        if resp.status == 429:
                            logger.warning(f"Rate limited for {instrument_name}, retry {attempt+1}")
                            await asyncio.sleep(5)
                            continue
                        resp.raise_for_status()
                        data = await resp.json()
                        return data.get('result', {})
                except Exception as e:
                    logger.error(f"Fetch order book error for {instrument_name}: {e}")
                    await asyncio.sleep(2)
            logger.error(f"Failed to fetch order book for {instrument_name} after 3 attempts")
            return {}

    async def fetch_and_store_option_chains(self):
        instruments = await self.fetch_option_instruments()
        btc_usd_price = self.fetch_btc_to_usd()
        if not instruments:
            logger.error("No instruments fetched; skipping option chains update")
            return

        async with aiohttp.ClientSession() as session_http:
            order_books = await asyncio.gather(*[
                self.fetch_order_book(session_http, inst['instrument_name']) for inst in instruments
            ])

        # Prepare data for probability calculation
        options_data = []
        for inst, ob in zip(instruments, order_books):
            if not ob:
                continue
            expiration = datetime.utcfromtimestamp(inst['expiration_timestamp'] / 1000).date()

            # Match the exact column names used in your probability function
            options_data.append({
                "Instrument": inst['instrument_name'],
                "Option Type": inst.get('option_type'),
                "Strike Price": inst.get('strike', 0),
                "Expiration Date": expiration,
                "Bid IV": ob.get('bid_iv', 0.0),
                "Ask IV": ob.get('ask_iv', 0.0),
                "Delta": ob.get('greeks', {}).get('delta'),
                "Gamma": ob.get('greeks', {}).get('gamma'),
                "Theta": ob.get('greeks', {}).get('theta'),
                "Vega": ob.get('greeks', {}).get('vega'),
                # These aren't used in probability but keep for your DB later:
                "Last Price (USD)": ob.get('last_price', 0.0) * btc_usd_price if ob.get('last_price') else 0.0,
                "Bid Price (USD)": ob.get('best_bid_price', 0.0) * btc_usd_price if ob.get('best_bid_price') else 0.0,
                "Ask Price (USD)": ob.get('best_ask_price', 0.0) * btc_usd_price if ob.get('best_ask_price') else 0.0,
                "Open Interest": ob.get('open_interest', 0.0),
                "Total Traded Volume": ob.get('stats', {}).get('volume', 0.0),
                "Monetary Volume": ob.get('stats', {}).get('volume_usd', 0.0),
            })

        # Create DataFrame and calculate probabilities
        df = pd.DataFrame(options_data)

        if not df.empty:
            prob_df = self.option_probabilities_with_greeks(df)
            prob_dict = prob_df.set_index('Instrument')['Exercise_Probability (%)'].to_dict()
        else:
            prob_dict = {}

        # Save to DB with probabilities
        def db_save():
            session = SessionLocal()
            count = 0
            try:
                for inst, ob in zip(instruments, order_books):
                    if not ob:
                        continue
                    expiration = datetime.utcfromtimestamp(inst['expiration_timestamp'] / 1000).date()
                    existing = session.query(OptionChain).filter_by(Instrument=inst['instrument_name']).first()

                    instrument_name = inst['instrument_name']
                    probability = prob_dict.get(instrument_name, 0.0)

                    data = {
                        "Instrument": instrument_name,
                        "Option_Type": inst.get('option_type'),
                        "Strike_Price": inst.get('strike', 0),
                        "Expiration_Date": expiration,
                        "Last_Price_USD": ob.get('last_price', 0.0) * btc_usd_price if ob.get('last_price') else 0.0,
                        "Bid_Price_USD": ob.get('best_bid_price', 0.0) * btc_usd_price if ob.get('best_bid_price') else 0.0,
                        "Ask_Price_USD": ob.get('best_ask_price', 0.0) * btc_usd_price if ob.get('best_ask_price') else 0.0,
                        "Bid_IV": ob.get('bid_iv', 0.0),
                        "Ask_IV": ob.get('ask_iv', 0.0),
                        "Delta": ob.get('greeks', {}).get('delta'),
                        "Gamma": ob.get('greeks', {}).get('gamma'),
                        "Theta": ob.get('greeks', {}).get('theta'),
                        "Vega": ob.get('greeks', {}).get('vega'),
                        "Open_Interest": ob.get('open_interest', 0.0),
                        "Total_Traded_Volume": ob.get('stats', {}).get('volume', 0.0),
                        "Monetary_Volume": ob.get('stats', {}).get('volume_usd', 0.0),
                        "Probability_Percent": probability,
                        "Timestamp": datetime.utcnow()
                    }

                    if existing:
                        for k, v in data.items():
                            setattr(existing, k, v)
                    else:
                        session.add(OptionChain(**data))

                    count += 1

                session.commit()
                logger.info(f"Saved {count} option chains.")
            except Exception as e:
                logger.error(f"Error saving option chains: {e}")
                session.rollback()
            finally:
                session.close()

        await asyncio.get_running_loop().run_in_executor(self.executor, db_save)


    async def fetch_public_trades_for_instrument(self, session_http, instrument, start_ts, end_ts):
        url = 'https://www.deribit.com/api/v2/public/get_last_trades_by_instrument_and_time'
        params = {
            'instrument_name': instrument,
            'start_timestamp': start_ts,
            'end_timestamp': end_ts,
            'count': 1000
        }

        async with self.semaphore:
            for attempt in range(5):  # more retries just in case
                try:
                    async with session_http.get(url, params=params) as resp:
                        if resp.status == 429:
                            delay = 2 ** attempt  # exponential backoff: 1, 2, 4, 8, 16 sec
                            logger.warning(f"Rate limited for {instrument}, retry {attempt + 1}, waiting {delay}s")
                            await asyncio.sleep(delay)
                            continue
                        resp.raise_for_status()
                        data = await resp.json()
                        return data.get('result', {}).get('trades', [])
                except aiohttp.ClientResponseError as e:
                    logger.error(f"Client error fetching trades for {instrument}: {e}")
                except aiohttp.ClientError as e:
                    logger.error(f"Connection error for {instrument}: {e}")
                except Exception as e:
                    logger.error(f"Unexpected error fetching trades for {instrument}: {e}")
                
                await asyncio.sleep(2)  # fallback wait on generic failure

            logger.error(f"Failed to fetch trades for {instrument} after multiple attempts.")
            return []

    def parse_instrument_metadata(self, instrument_name):
        try:
            parts = instrument_name.split('-')
            if len(parts) != 4:
                return None, None, None

            _, date_str, strike, option_type_code = parts
            expiration_date = datetime.strptime(date_str, "%d%b%y").date()
            strike_price = float(strike)

            # Map option type
            option_type = 'Call' if option_type_code.upper() == 'C' else 'Put' if option_type_code.upper() == 'P' else None

            return expiration_date, strike_price, option_type
        except Exception as e:
            logger.warning(f"Failed to parse instrument metadata for {instrument_name}: {e}")
            return None, None, None

    async def fetch_and_store_public_trades(self):
        session = SessionLocal()
        instruments = [i.Instrument for i in session.query(OptionChain.Instrument).all()]
        session.close()

        end = datetime.utcnow()
        start = end - timedelta(hours=24)
        start_ts = int(start.timestamp() * 1000)
        end_ts = int(end.timestamp() * 1000)

        async with aiohttp.ClientSession() as session_http:
            all_trades = await asyncio.gather(*[
                self.fetch_public_trades_for_instrument(session_http, inst, start_ts, end_ts) for inst in instruments
            ])

        def db_save():
            session = SessionLocal()
            count = 0
            try:
                self.remove_expired_trades_from_db(session)
                for trades in all_trades:
                    for t in trades:
                        try:
                            trade_id = str(t['trade_id'])
                            if session.query(PublicTrade).filter_by(Trade_ID=trade_id).first():
                                continue

                            expiration_date, strike_price, option_type = self.parse_instrument_metadata(t.get('instrument_name'))

                            new_trade = PublicTrade(
                                Trade_ID=trade_id,
                                Side=t.get('direction'),
                                Instrument=t.get('instrument_name'),
                                Price_BTC=t.get('price'),
                                Price_USD=t.get('price') * t.get('index_price'),
                                IV_Percent=t.get('iv'),
                                Size=t.get('amount'),
                                Entry_Value=t['amount'] * t['price'] * t['index_price'],
                                Underlying_Price=t.get('index_price'),
                                Expiration_Date=expiration_date,
                                Strike_Price=strike_price,
                                Option_Type=option_type,
                                Entry_Date=datetime.utcfromtimestamp(t['timestamp'] / 1000),
                                BlockTrade_IDs=','.join(t.get('block_trade_id', [])) if 'block_trade_id' in t else None,
                                BlockTrade_Count=len(t.get('block_trade_id', [])) if 'block_trade_id' in t else None,
                                Combo_ID=t.get('combo_id'),
                                ComboTrade_IDs=','.join(t.get('combo_trade_id', [])) if 'combo_trade_id' in t else None,
                            )
                            session.add(new_trade)
                            count += 1
                        except IntegrityError:
                            session.rollback()
                        except Exception as e:
                            logger.error(f"Error saving trade {t.get('trade_id')}: {e}")
                            session.rollback()
                session.commit()
                logger.info(f"Saved {count} public trades.")
            except Exception as e:
                logger.error(f"Error saving public trades: {e}")
                session.rollback()
            finally:
                session.close()

        await asyncio.get_running_loop().run_in_executor(self.executor, db_save)
        
    def remove_expired_trades_from_db(self, session):
        current_utc_datetime = datetime.utcnow()
        if current_utc_datetime.hour >= 8:
            cutoff_date = current_utc_datetime.date()
        else:
            cutoff_date = (current_utc_datetime - timedelta(days=1)).date()
        cutoff_datetime = datetime.combine(cutoff_date, datetime.min.time())

        num_deleted = session.query(PublicTrade) \
            .filter(PublicTrade.Expiration_Date < cutoff_datetime) \
            .delete(synchronize_session=False)
        logger.info(f"Removed {num_deleted} expired public trades from DB before storing new trades.")
    
    def option_probabilities_with_greeks(self, df):
        df['Expiration Date'] = pd.to_datetime(df['Expiration Date'])
        df['Option Type'] = LabelEncoder().fit_transform(df['Option Type'])
        df['Time to Expiration'] = (df['Expiration Date'] - pd.to_datetime('today')).dt.days

        delta_threshold = 0.5
        df['Exercise'] = (df['Delta'] > delta_threshold).astype(int)

        features = ['Option Type', 'Strike Price', 'Bid IV', 'Ask IV', 'Gamma', 'Theta', 'Vega', 'Time to Expiration']
        target = 'Exercise'
        
        X = df[features]
        y = df[target]

        df_majority = df[df.Exercise == 0]
        df_minority = df[df.Exercise == 1]
        df_minority_upsampled = resample(df_minority, replace=True, n_samples=len(df_majority), random_state=42)
        df_balanced = pd.concat([df_majority, df_minority_upsampled])

        X_balanced = df_balanced[features]
        y_balanced = df_balanced[target]

        X_train, X_valid, y_train, y_valid = train_test_split(X_balanced, y_balanced, test_size=0.2, random_state=42, stratify=y_balanced)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_valid_scaled = scaler.transform(X_valid)

        # Convert to DataFrame to maintain feature names
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=features)
        X_valid_scaled = pd.DataFrame(X_valid_scaled, columns=features)

        xgb_search = BayesSearchCV(
            xgb.XGBClassifier(objective='binary:logistic', use_label_encoder=False, verbosity=0),
            {'n_estimators': (50, 200), 'learning_rate': (0.01, 0.1), 'max_depth': (3, 10)},
            n_iter=20,
            cv=3,
            scoring='accuracy'
        )
        xgb_search.fit(X_train_scaled, y_train)
        best_xgb_model = xgb_search.best_estimator_

        lgb_model = lgb.LGBMClassifier(objective='binary', metric='binary_logloss', verbose=-1)
        lgb_search = BayesSearchCV(
            lgb_model,
            {'num_leaves': (31, 100), 'learning_rate': (0.01, 0.1), 'n_estimators': (50, 200)},
            n_iter=20,
            cv=3,
            scoring='accuracy'
        )

        lgb_search.fit(X_train_scaled, y_train)
        best_lgb_model = lgb_search.best_estimator_

        voting_model = VotingClassifier(estimators=[('xgb', best_xgb_model), ('lgbm', best_lgb_model)], voting='soft')
        voting_model.fit(X_train_scaled, y_train)

        y_proba = voting_model.predict_proba(X_valid_scaled)[:, 1]

        results_df = df.iloc[y_valid.index].copy()
        results_df['Exercise_Probability (%)'] = y_proba * 100

        # Modify the Exercise Probability
        results_df['Exercise_Probability (%)'] = results_df['Exercise_Probability (%)'].apply(lambda x: int(round((x - int(x)) * 1000)) / 10)

        return results_df[['Instrument', 'Exercise_Probability (%)']]
    
    def fetch_btc_to_usd(self):
        """Fetch current BTC to USD conversion rate.""" 
        
        url = "https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd"
        try:
            response = self.session.get(url)
            response.raise_for_status()
            data = response.json()
            self.btc_usd_price = data.get('bitcoin', {}).get('usd', 0)
            return self.btc_usd_price
        except requests.RequestException as e:
            logging.error(f"Error fetching BTC price: {e}")
            return 0
        
    def fetch_today_high_low(self):
        # Using get_book_summary_by_currency which reliably returns high/low for BTC
        url = "https://www.deribit.com/api/v2/public/get_book_summary_by_currency"
        params = {
            'currency': 'BTC',
            'kind': 'future'  # For perpetual swaps and futures
        }
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            if not data.get('result'):
                print("Error: No results in getting highest and lowest price response")
                return None, None
                
            # Find BTC-PERPETUAL in results
            for instrument in data['result']:
                if instrument['instrument_name'] == 'BTC-PERPETUAL':
                    highest_price = int(float(instrument['high']))
                    lowest_price = int(float(instrument['low']))
                    return highest_price, lowest_price
            
            return None, None
            
        except requests.exceptions.RequestException as e:
            print(f"API Request Failed: {e}")
            return None, None
        except (KeyError, ValueError) as e:
            print(f"Data Parsing Error: {e}")
            return None, None