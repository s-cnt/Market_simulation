import json
import time
import asyncio
import websockets
import logging
from datetime import datetime

class OrderbookWebsocketClient:
    def __init__(self, url, symbol, callback):
        self.url = url
        self.symbol = symbol
        self.callback = callback
        self.logger = logging.getLogger("websocket_client")
        self.is_connected = False
        self.should_continue = True
        self.reconnect_delay = 1  # Start with 1 second delay
        self.max_reconnect_delay = 60  # Maximum 60 seconds between retries
        
    async def connect(self):
        try:
            self.ws = await websockets.connect(self.url)
            self.is_connected = True
            self.reconnect_delay = 1  # Reset delay on successful connection
            self.logger.info(f"Connected to {self.url}")
            return True
        except Exception as e:
            self.logger.error(f"Connection error: {str(e)}")
            return False
            
    async def disconnect(self):
        self.should_continue = False
        if self.is_connected:
            await self.ws.close()
            self.is_connected = False
            self.logger.info("Disconnected from WebSocket")
            
    async def listen(self):
        while self.should_continue:
            if not self.is_connected:
                success = await self.connect()
                if not success:
                    # Exponential backoff for reconnection attempts
                    self.logger.info(f"Reconnecting in {self.reconnect_delay} seconds...")
                    await asyncio.sleep(self.reconnect_delay)
                    self.reconnect_delay = min(self.reconnect_delay * 2, self.max_reconnect_delay)
                    continue
                    
            try:
                message = await self.ws.recv()
                start_time = time.time()
                data = json.loads(message)
                self.logger.info(f"Received orderbook: {data}")
                # Process the data
                await self.callback(data, start_time)
                
            except websockets.exceptions.ConnectionClosed:
                self.logger.warning("Connection closed unexpectedly")
                self.is_connected = False
                continue
                
            except Exception as e:
                self.logger.error(f"Error processing message: {str(e)}")
                # Continue receiving messages even if processing fails
                continue
                
        self.logger.info("WebSocket listener stopped")