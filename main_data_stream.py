import asyncio
import logging
from deribit_client import DeribitClient
from config import DERIBIT_CLIENT_ID, DERIBIT_CLIENT_SECRET
from db import init_db, get_latest_option_chains, get_latest_public_trades

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def run_continuous(client, interval_seconds=60):
    while True:
        try:
            logger.info("Starting option chains fetch & store...")
            await client.fetch_and_store_option_chains()
            logger.info("Option chains fetched and stored.")

            logger.info("Starting public trades fetch & store...")
            await client.fetch_and_store_public_trades()
            logger.info("Public trades fetched and stored.")
        except Exception as e:
            logger.error(f"Error during fetch/store cycle: {e}")
        await asyncio.sleep(interval_seconds)

async def main():
    init_db()
    client = DeribitClient(DERIBIT_CLIENT_ID, DERIBIT_CLIENT_SECRET)
    await run_continuous(client)



if __name__ == "__main__":
    asyncio.run(main())
