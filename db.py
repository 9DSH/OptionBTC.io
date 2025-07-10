## db.py
import logging
from sqlalchemy import Column, Integer, Float, String, Date, DateTime, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from config import DB_URI

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

Base = declarative_base()

class OptionChain(Base):
    __tablename__ = 'option_chains'
    id = Column(Integer, primary_key=True, autoincrement=True)
    Instrument = Column(String, unique=True, index=True, nullable=False)  # instrument name like BTC-29JUN25-96000-C
    Option_Type = Column(String)
    Strike_Price = Column(Float)
    Expiration_Date = Column(Date)
    Last_Price_USD = Column(Float)
    Bid_Price_USD = Column(Float)
    Ask_Price_USD = Column(Float)
    Bid_IV = Column(Float)
    Ask_IV = Column(Float)
    Delta = Column(Float)
    Gamma = Column(Float)
    Theta = Column(Float)
    Vega = Column(Float)
    Open_Interest = Column(Float)
    Total_Traded_Volume = Column(Float)
    Monetary_Volume = Column(Float)
    Probability_Percent = Column(Float)
    Timestamp = Column(DateTime)

class PublicTrade(Base):
    __tablename__ = 'public_trades'
    Trade_ID = Column(String, primary_key=True, index=True)  # unique trade ID
    Side = Column(String)
    Instrument = Column(String)
    Price_BTC = Column(Float)
    Price_USD = Column(Float)
    IV_Percent = Column(Float)
    Size = Column(Float)
    Entry_Value = Column(Float)
    Underlying_Price = Column(Float)
    Expiration_Date = Column(Date)
    Strike_Price = Column(Float)
    Option_Type = Column(String)
    Entry_Date = Column(DateTime)
    BlockTrade_IDs = Column(String)       
    BlockTrade_Count = Column(Integer)
    Combo_ID = Column(String)
    ComboTrade_IDs = Column(String)       

engine = create_engine(DB_URI, connect_args={"check_same_thread": False} if DB_URI.startswith('sqlite') else {})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def init_db():
    Base.metadata.create_all(bind=engine)
    logger.info("Database initialized.")

def get_latest_option_chains(limit=5):
    session = SessionLocal()
    try:
        return session.query(OptionChain).order_by(OptionChain.timestamp.desc()).limit(limit).all()
    finally:
        session.close()

def get_latest_public_trades(limit=5):
    session = SessionLocal()
    try:
        return session.query(PublicTrade).order_by(PublicTrade.entry_date.desc()).limit(limit).all()
    finally:
        session.close()
