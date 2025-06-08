import asyncio
import json
import logging
import time
from trade_simulator.websocket_client import OrderbookWebsocketClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger()

async def handle_orderbook(data, latency):
    logger.info(f"Received orderbook: {json.dumps(data, indent=2)}")
    logger.info(f"Processing latency: {latency*1000:.2f}ms")

async def main():
    # Create a modified client with direct message logging
    class DebugWebsocketClient(OrderbookWebsocketClient):
        async def listen(self):
            while self.is_connected:
                try:
                    message = await self.ws.recv()
                    print(f"RAW MESSAGE: {message}")  # Print raw message
                    start_time = time.time()
                    data = json.loads(message)
                    await self.callback(data, start_time)
                except Exception as e:
                    print(f"Error in listen: {e}")
                    continue

    # Use the endpoint specified in the assignment
    client = DebugWebsocketClient(
        url="wss://ws.gomarket-cpp.goquant.io/ws/l2-orderbook/okx/BTC-USDT-SWAP",
        symbol="BTC-USDT-SWAP", 
        callback=handle_orderbook
    )
    
    try:
        await client.connect()
        # No need for subscription, this endpoint likely streams automatically
        # Keep the connection open for 60 seconds
        await asyncio.sleep(60)
    finally:
        await client.disconnect()

if __name__ == "__main__":
    asyncio.run(main())