# realtime_simulator.py - FIXED VERSION
import asyncio
import websockets
import json
import random
from datetime import datetime

class RealTimeHazardSimulator:
    def _init_(self):
        self.connected_clients = set()
        self.hazard_data = self.initialize_hazard_data()
    
    def initialize_hazard_data(self):
        return {
            'heat_risk': {'current': 65, 'trend': 'increasing'},
            'air_quality': {'current_aqi': 156, 'pm25': 68},
            'flood_risk': {'current': 28, 'rainfall': 12.5}
        }
    
    def generate_live_updates(self):
        self.hazard_data['heat_risk']['current'] = max(0, min(100, 
            self.hazard_data['heat_risk']['current'] + random.uniform(-2, 3)))
        self.hazard_data['air_quality']['pm25'] = max(10, min(300, 
            self.hazard_data['air_quality']['pm25'] + random.uniform(-5, 8)))
        return self.hazard_data
    
    async def handle_client(self, websocket, path):  # FIXED: Added path parameter
        self.connected_clients.add(websocket)
        try:
            await websocket.send(json.dumps({
                'type': 'initial_data',
                'data': self.hazard_data,
                'timestamp': datetime.now().isoformat()
            }))
            
            async for message in websocket:
                if message == 'request_update':
                    update = self.generate_live_updates()
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
                update = self.generate_live_updates()
                message = json.dumps({
                    'type': 'broadcast',
                    'data': update,
                    'timestamp': datetime.now().isoformat()
                })
                await asyncio.gather(
                    *[client.send(message) for client in self.connected_clients],
                    return_exceptions=True
                )
            await asyncio.sleep(5)

async def main():
    simulator = RealTimeHazardSimulator()
    server = await websockets.serve(simulator.handle_client, "localhost", 8765)
    asyncio.create_task(simulator.broadcast_updates())
    
    print("Real-time server running on ws://localhost:8765")
    await server.wait_closed()

if _name_ == "_main_":
    asyncio.run(main())