# realtime_simulator.py - FIXED VERSION
import asyncio
import websockets
import json
import random
from datetime import datetime
import requests # Make sure to install this library
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("OPENAPI_KEY")

class RealTimeHazardSimulator:
    def __init__(self):
        self.connected_clients = set()
        self.api_key = api_key 
        self.chennai_lat = 13.0827
        self.chennai_lon = 80.2707

    def get_live_updates(self):
        """Fetches real-time weather and air quality data."""
        weather_url = f"http://api.openweathermap.org/data/2.5/weather?lat={self.chennai_lat}&lon={self.chennai_lon}&appid={self.api_key}&units=metric"
        air_pollution_url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={self.chennai_lat}&lon={self.chennai_lon}&appid={self.api_key}"

        weather_response = requests.get(weather_url)
        air_pollution_response = requests.get(air_pollution_url)

        if weather_response.status_code == 200 and air_pollution_response.status_code == 200:
            weather_data = weather_response.json()
            air_pollution_data = air_pollution_response.json()

            return {
                'heat_risk': {'current': weather_data['main']['temp'], 'trend': 'stable'},
                'air_quality': {'current_aqi': air_pollution_data['list'][0]['main']['aqi'], 'pm25': air_pollution_data['list'][0]['components']['pm2_5']},
                'flood_risk': {'current': 28, 'rainfall': weather_data.get('rain', {}).get('1h', 0)} # flood risk needs a more complex model
            }
        else:
            # Fallback to simulated data if API fails
            print("APi failed, using simulated data")
            return {
                'heat_risk': {'current': 32 + random.uniform(-2, 2), 'trend': 'increasing'},
                'air_quality': {'current_aqi': 150 + random.uniform(-10, 10), 'pm25': 60 + random.uniform(-5, 5)},
                'flood_risk': {'current': 28 + random.uniform(-5, 5), 'rainfall': 12.5 + random.uniform(-2, 2)}
            }


    async def handle_client(self, websocket, path):  # FIXED: Added path parameter
        self.connected_clients.add(websocket)
        try:
            await websocket.send(json.dumps({
                'type': 'initial_data',
                'data': self.get_live_updates(),
                'timestamp': datetime.now().isoformat()
            }))

            async for message in websocket:
                if message == 'request_update':
                    update = self.get_live_updates()
                    await websocket.send(json.dumps({
                        'type': 'update',
                        'data': update,
                        'timestamp': datetime.now().isoformat()
                    }))

        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            self.connected_clients.remove(websocket)

    async def broadcast_updates(self):
        while True:
            if self.connected_clients:
                update = self.get_live_updates()
                message = json.dumps({
                    'type': 'broadcast',
                    'data': update,
                    'timestamp': datetime.now().isoformat()
                })
                await asyncio.gather(
                    *[client.send(message) for client in self.connected_clients],
                    return_exceptions=True
                )
            await asyncio.sleep(60) # Fetch new data every 60 seconds

async def main():
    simulator = RealTimeHazardSimulator()
    server = await websockets.serve(simulator.handle_client, "localhost", 8765)
    asyncio.create_task(simulator.broadcast_updates())

    print("Real-time server running on ws://localhost:8765")
    await server.wait_closed()

if __name__ == "__main__":
    asyncio.run(main())