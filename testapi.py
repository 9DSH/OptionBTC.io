
import requests
import json
from datetime import datetime, timedelta, timezone
import argparse
import time

# Deribit public API endpoint for fetching last trades by instrument and time
deribit_trades_url = (
    'https://www.deribit.com/api/v2/public/get_last_trades_by_instrument_and_time'
)


def fetch_public_trades(instrument_name: str,
                        start_timestamp: int,
                        end_timestamp: int,
                        count: int = 1000) -> list:
    """
    Fetch the most recent public trades for a Deribit instrument between two timestamps.

    :param instrument_name: The Deribit instrument name, e.g., 'BTC-PERPETUAL'.
    :param start_timestamp: Start time in milliseconds since epoch (inclusive).
    :param end_timestamp: End time in milliseconds since epoch (exclusive).
    :param count: Number of trades to return (max 1000 per request).
    :return: A list of trade objects.
    """
    params = {
        'instrument_name': instrument_name,
        'start_timestamp': start_timestamp,
        'end_timestamp': end_timestamp,
        'count': count
    }

    response = requests.get(deribit_trades_url, params=params)
    response.raise_for_status()
    data = response.json()

    return data.get('result', {}).get('trades', [])


def main():
    parser = argparse.ArgumentParser(
        description='Fetch public trades from Deribit API for a specified instrument.'
    )
    parser.add_argument(
        '--instrument', '-i',
        type=str,
        default='BTC-PERPETUAL',
        help="Deribit instrument name (e.g., 'BTC-PERPETUAL' or 'BTC-30JUN25-60000-C')"
    )
    parser.add_argument(
        '--hours', '-H',
        type=int,
        default=24,
        help='Number of hours back from now to fetch trades.'
    )
    args = parser.parse_args()

    instrument = args.instrument
    now_utc = datetime.now(timezone.utc)
    start_utc = now_utc - timedelta(hours=args.hours)

    start_ms = int(start_utc.timestamp() * 1000)
    end_ms = int(now_utc.timestamp() * 1000)

    print(
        f"Fetching public trades for {instrument} "
        f"from {start_utc.isoformat()} to {now_utc.isoformat()}..."
    )

    try:
        trades = fetch_public_trades(instrument, start_ms, end_ms)
        print(f"Retrieved {len(trades)} trades.")

        output_file = f"public_trades_{instrument}_{int(time.time())}.json"
        with open(output_file, 'w') as f:
            json.dump(trades, f, indent=2)

        print(f"Saved trades to {output_file}")
    except Exception as e:
        print(f"Error fetching trades: {e}")


if __name__ == '__main__':
    main()

