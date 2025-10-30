# emergency_response.py
import json
import smtplib
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart
import requests

class EmergencyAlertSystem:
    def __init__(self):
        self.alert_levels = {
            'low': {'color': 'green', 'priority': 1},
            'moderate': {'color': 'yellow', 'priority': 2},
            'high': {'color': 'orange', 'priority': 3},
            'critical': {'color': 'red', 'priority': 4}
        }
    
    def assess_emergency_level(self, risk_data):
        """Assess emergency level based on risk data"""
        overall_risk = risk_data.get('overall_risk', 0)
        
        if overall_risk >= 90:
            return 'critical'
        elif overall_risk >= 70:
            return 'high'
        elif overall_risk >= 40:
            return 'moderate'
        else:
            return 'low'
    
    def generate_alert_message(self, risk_data, emergency_level):
        """Generate appropriate alert message"""
        templates = {
            'critical': """
            🚨 CRITICAL ALERT - IMMEDIATE ACTION REQUIRED 🚨
            
            Location: {location}
            Risk Level: CRITICAL ({risk_score}%)
            
            Hazards Detected:
            - Heat Risk: {heat_risk}%
            - Air Quality Risk: {air_risk}%
            - Flood Risk: {flood_risk}%
            
            Recommended Actions:
            • Evacuate immediately if in flood zone
            • Seek air-conditioned shelter
            • Use N95 masks outdoors
            • Monitor official channels
            
            Emergency services have been notified.
            """,
            
            'high': """
            ⚠️ HIGH RISK ALERT - PRECAUTIONS ADVISED ⚠️
            
            Location: {location}
            Risk Level: HIGH ({risk_score}%)
            
            Hazards Detected:
            - Heat Risk: {heat_risk}%
            - Air Quality Risk: {air_risk}%
            - Flood Risk: {flood_risk}%
            
            Recommended Actions:
            • Limit outdoor activities
            • Stay hydrated
            • Monitor weather updates
            • Prepare emergency kit
            """
        }
        
        template = templates.get(emergency_level, "No alert necessary")
        return template.format(
            location=risk_data.get('location', 'Unknown'),
            risk_score=risk_data.get('overall_risk', 0),
            heat_risk=risk_data.get('heat_risk', 0),
            air_risk=risk_data.get('air_quality_risk', 0),
            flood_risk=risk_data.get('flood_risk', 0)
        )
    
    def send_alerts(self, risk_data):
        """Send alerts through multiple channels"""
        emergency_level = self.assess_emergency_level(risk_data)
        
        if emergency_level in ['high', 'critical']:
            alert_message = self.generate_alert_message(risk_data, emergency_level)
            
            # Simulate sending alerts (in real implementation, integrate with actual services)
            print(f"🔔 SENDING {emergency_level.upper()} ALERT:")
            print(alert_message)
            print("-" * 50)
            
            # Log alert
            self.log_alert(risk_data, emergency_level, alert_message)
            
            return True
        return False
    
    def log_alert(self, risk_data, level, message):
        """Log alert to file"""
        alert_log = {
            'timestamp': risk_data.get('timestamp'),
            'location': risk_data.get('location'),
            'emergency_level': level,
            'risk_scores': {
                'overall': risk_data.get('overall_risk'),
                'heat': risk_data.get('heat_risk'),
                'air_quality': risk_data.get('air_quality_risk'),
                'flood': risk_data.get('flood_risk')
            },
            'message': message
        }
        
        try:
            with open('output/alert_log.json', 'a') as f:
                f.write(json.dumps(alert_log) + '\n')
        except Exception as e:
            print(f"Error logging alert: {e}")

class EvacuationPlanner:
    def __init__(self):
        self.shelter_locations = self.load_shelters()
    
    def load_shelters(self):
        """Load emergency shelter locations"""
        return [
            {'id': 1, 'name': 'Chennai Central Station', 'lat': 13.0827, 'lon': 80.2707, 'capacity': 500},
            {'id': 2, 'name': 'Anna University', 'lat': 13.0102, 'lon': 80.2357, 'capacity': 300},
            {'id': 3, 'name': 'Marina Beach Complex', 'lat': 13.0500, 'lon': 80.2820, 'capacity': 200}
        ]
    
    def plan_evacuation_routes(self, risk_zones):
        """Generate evacuation routes from risk zones"""
        routes = {}
        
        for zone in risk_zones:
            # Find nearest shelter
            nearest_shelter = min(self.shelter_locations, 
                                key=lambda s: self.calculate_distance(zone, s))
            
            routes[zone['id']] = {
                'from': zone,
                'to': nearest_shelter,
                'distance_km': self.calculate_distance(zone, nearest_shelter),
                'estimated_time': self.estimate_travel_time(zone, nearest_shelter)
            }
        
        return routes
    
    def calculate_distance(self, point1, point2):
        """Calculate distance between two points (simplified)"""
        lat_diff = abs(point1['lat'] - point2['lat'])
        lon_diff = abs(point1['lon'] - point2['lon'])
        return (lat_diff + lon_diff) * 111  # Approximate km
    
    def estimate_travel_time(self, point1, point2):
        """Estimate travel time in minutes"""
        distance = self.calculate_distance(point1, point2)
        return distance * 2  # Assuming 30 km/h average speed

# Test emergency system
def test_emergency_system():
    print("🚨 Testing Emergency Response System...")
    
    alert_system = EmergencyAlertSystem()
    evacuation_planner = EvacuationPlanner()
    
    # Test data
    test_risk = {
        'location': 'Central Chennai',
        'overall_risk': 85,
        'heat_risk': 90,
        'air_quality_risk': 70,
        'flood_risk': 60,
        'timestamp': '2024-01-15T14:30:00'
    }
    
    # Send alert
    alert_system.send_alerts(test_risk)
    
    # Test evacuation planning
    risk_zones = [
        {'id': 'zone1', 'lat': 13.0827, 'lon': 80.2707, 'population': 1000},
        {'id': 'zone2', 'lat': 13.0500, 'lon': 80.2820, 'population': 500}
    ]
    
    routes = evacuation_planner.plan_evacuation_routes(risk_zones)
    print("📍 Evacuation Routes Planned:")
    for zone_id, route in routes.items():
        print(f"Zone {zone_id} → {route['to']['name']}: {route['distance_km']:.1f} km")
    
    # Save evacuation plan
    with open('output/evacuation_plan.json', 'w') as f:
        json.dump(routes, f, indent=2)
    
    print("Emergency system tested successfully")

if __name__ == "__main__":
    test_emergency_system()