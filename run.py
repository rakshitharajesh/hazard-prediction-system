#!/usr/bin/env python3
"""
AI-Powered Urban Hazard Digital Twin - Fixed Version (No Emojis)
"""
import json
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
import joblib
import os
import requests
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("OPENAPI_KEY")

print("Initializing AI-Powered Urban Hazard Digital Twin...")
print("Focus Area: Chennai Metropolitan Area")
print("=" * 60)

# Create output directory
os.makedirs("output/charts", exist_ok=True)

class HazardPredictor:
    def __init__(self):
        self.chennai_center = (12.8406, 80.1534)
        self.grid_size = 20  # 20x20 = 400 locations
        self.api_key = api_key

    def generate_spatial_data(self):
        print("Generating spatial urban data...")
        
        lats = np.linspace(12.8, 13.3, self.grid_size)
        lons = np.linspace(80.0, 80.5, self.grid_size)
        
        spatial_data = []
        for i, lat in enumerate(lats):
            for j, lon in enumerate(lons):
                population_density = np.random.normal(15000, 5000)
                building_height = np.random.normal(25, 15)
                green_cover = np.random.uniform(5, 30)
                
                spatial_data.append({
                    'cell_id': f"{i}_{j}",
                    'latitude': lat,
                    'longitude': lon,
                    'population_density': max(1000, population_density),
                    'building_height': max(5, building_height),
                    'green_cover': green_cover,
                    'urban_heat_bias': np.random.uniform(0, 2),
                    'elevation': np.random.uniform(0, 50)
                })
        
        df = pd.DataFrame(spatial_data)
        print(f"Generated {len(df)} spatial locations")
        return df

    def get_realtime_weather(self, lat, lon):
        """Fetches real-time weather and air quality data"""
        try:
            # Get weather data
            weather_url = f"http://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={self.api_key}&units=metric"
            weather_response = requests.get(weather_url, timeout=5)
            
            # Get air pollution data
            air_pollution_url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={self.api_key}"
            air_pollution_response = requests.get(air_pollution_url, timeout=5)
            
            if weather_response.status_code == 200:
                weather_data = weather_response.json()
                
                # Get PM2.5 from air pollution API
                pm25 = 25.0  # default fallback
                if air_pollution_response.status_code == 200:
                    air_data = air_pollution_response.json()
                    if 'list' in air_data and len(air_data['list']) > 0:
                        pm25 = air_data['list'][0]['components']['pm2_5']
                
                # Estimate rainfall from weather condition
                weather_main = weather_data['weather'][0]['main'].lower()
                weather_desc = weather_data['weather'][0]['description'].lower()
                rainfall_estimate = self.estimate_rainfall_from_weather(weather_main, weather_desc)
                
                return {
                    'temperature': weather_data['main']['temp'],
                    'humidity': weather_data['main']['humidity'],
                    'rainfall_estimate': rainfall_estimate,
                    'pm25': pm25,
                    'wind_speed': weather_data['wind']['speed'],
                    'pressure': weather_data['main']['pressure'],
                    'cloud_coverage': weather_data['clouds']['all'],
                    'weather_condition': weather_main
                }
        except Exception as e:
            print(f"Weather API error: {e}")
        
        # Fallback: generate synthetic weather data
        return self.generate_synthetic_weather_data()

    def estimate_rainfall_from_weather(self, weather_main, weather_desc):
        """Estimate rainfall intensity based on weather description"""
        # Rainfall estimation in mm/h based on weather conditions
        if 'heavy' in weather_desc or 'torrential' in weather_desc:
            return np.random.uniform(15, 30)  # Heavy rain
        elif 'moderate' in weather_desc:
            return np.random.uniform(5, 15)   # Moderate rain
        elif 'light' in weather_desc or 'drizzle' in weather_desc:
            return np.random.uniform(1, 5)    # Light rain/drizzle
        elif any(keyword in weather_main for keyword in ['rain', 'thunderstorm']):
            return np.random.uniform(8, 20)   # General rain
        elif 'shower' in weather_desc:
            return np.random.uniform(3, 10)   # Showers
        else:
            return 0.0  # No rain

    def generate_synthetic_weather_data(self):
        """Generate synthetic weather data when API fails"""
        hour = datetime.now().hour
        base_temp = 28 + 8 * np.sin(2 * np.pi * (hour - 14) / 24)
        
        # Chennai monsoon season (Oct-Dec) has higher rainfall probability
        current_month = datetime.now().month
        monsoon_season = current_month in [10, 11, 12]
        base_rain_prob = 0.4 if monsoon_season else 0.1
        
        # Generate rainfall based on probability
        rainfall = np.random.exponential(5) if np.random.random() < base_rain_prob else 0
        
        return {
            'temperature': max(20, min(40, base_temp + np.random.normal(0, 2))),
            'humidity': np.random.uniform(60, 90),
            'rainfall_estimate': rainfall,
            'pm25': max(10, np.random.normal(45, 20)),
            'wind_speed': np.random.uniform(0, 12),
            'pressure': np.random.normal(1010, 8),
            'cloud_coverage': np.random.uniform(0, 100),
            'weather_condition': 'clouds'
        }

    def simulate_weather_conditions(self, spatial_df):
        print("Generating weather conditions for all locations...")
        
        weather_data = []
        
        for idx, location in spatial_df.iterrows():
            data = self.get_realtime_weather(location['latitude'], location['longitude'])
            
            weather_data.append({
                'timestamp': datetime.now().isoformat(),
                'cell_id': location['cell_id'],
                'temperature': data['temperature'],
                'humidity': data['humidity'],
                'rainfall_estimate': data['rainfall_estimate'],
                'pm25': data['pm25'],
                'wind_speed': data['wind_speed'],
                'pressure': data['pressure'],
                'cloud_coverage': data['cloud_coverage'],
                'population_density': location['population_density'],
                'green_cover': location['green_cover'],
                'building_height': location['building_height']
            })
            
            # Progress indicator
            if (idx + 1) % 50 == 0:
                print(f"   Processed {idx + 1}/{len(spatial_df)} locations")
        
        df = pd.DataFrame(weather_data)
        print(f"Generated weather data for {len(df)} locations")
        return df

    def calculate_heat_index(self, temp, humidity):
        hi = temp + 0.5 * (temp - 20) * (humidity - 50) / 50
        return max(temp, hi)

    def calculate_air_quality_index(self, pm25):
        """Convert PM2.5 to proper Air Quality Index (AQI)"""
        # US EPA AQI scale for PM2.5
        if pm25 <= 12.0:
            aqi = ((50 - 0) / (12.0 - 0.0)) * (pm25 - 0.0) + 0
        elif pm25 <= 35.4:
            aqi = ((100 - 51) / (35.4 - 12.1)) * (pm25 - 12.1) + 51
        elif pm25 <= 55.4:
            aqi = ((150 - 101) / (55.4 - 35.5)) * (pm25 - 35.5) + 101
        elif pm25 <= 150.4:
            aqi = ((200 - 151) / (150.4 - 55.5)) * (pm25 - 55.5) + 151
        elif pm25 <= 250.4:
            aqi = ((300 - 201) / (250.4 - 150.5)) * (pm25 - 150.5) + 201
        else:
            aqi = ((500 - 301) / (500.4 - 250.5)) * (pm25 - 250.5) + 301
            
        return min(500, max(0, aqi))

    def calculate_air_quality_risk(self, aqi):
        """Convert AQI to risk percentage (0-100%)"""
        if aqi <= 50:
            return 0  # Good
        elif aqi <= 100:
            return (aqi - 50) * 0.5  # Moderate (0-25%)
        elif aqi <= 150:
            return 25 + (aqi - 100) * 0.5  # Unhealthy for sensitive groups (25-50%)
        elif aqi <= 200:
            return 50 + (aqi - 150) * 0.5  # Unhealthy (50-75%)
        elif aqi <= 300:
            return 75 + (aqi - 200) * 0.25  # Very Unhealthy (75-100%)
        else:
            return 100  # Hazardous

    def calculate_flood_risk(self, location, weather_data):
        """Comprehensive flood risk calculation"""
        lat, lon = location['latitude'], location['longitude']
        
        # 1. Rainfall factor (main driver)
        rainfall_factor = min(1.0, weather_data['rainfall_estimate'] / 30)  # Normalize by 30mm/h
        
        # 2. Topography and elevation (Chennai is largely flat coastal)
        elevation_risk = 1 - (location['elevation'] / 100)  # Lower elevation = higher risk
        elevation_risk = max(0.1, min(1.0, elevation_risk))
        
        # 3. Urban drainage capacity (based on population density)
        # Higher density = worse drainage = higher flood risk
        drainage_factor = min(1.0, location['population_density'] / 25000)
        
        # 4. Proximity to water bodies (Chennai-specific)
        water_proximity_risk = self.calculate_water_proximity_risk(lat, lon)
        
        # 5. Soil saturation (simulated based on recent weather)
        soil_saturation = 0.3 + (weather_data['humidity'] / 200)  # Higher humidity = more saturated
        
        # 6. Green cover factor (more green cover = better absorption)
        green_cover_factor = 1 - (location['green_cover'] / 50)  # More green = lower risk
        
        # Combined flood risk calculation with weights
        flood_risk = (
            rainfall_factor * 0.35 +
            elevation_risk * 0.20 +
            drainage_factor * 0.15 +
            water_proximity_risk * 0.15 +
            soil_saturation * 0.10 +
            green_cover_factor * 0.05
        )
        
        # Chennai flood-prone areas adjustment
        if self.is_flood_prone_area(lat, lon):
            flood_risk = min(1.0, flood_risk * 1.3)
        
        return min(100, flood_risk * 100)

    def calculate_water_proximity_risk(self, lat, lon):
        """Calculate risk based on proximity to water bodies in Chennai"""
        # Chennai water bodies coordinates (approximate)
        water_bodies = [
            (13.0418, 80.2341),  # Adyar River
            (13.1673, 80.3008),  # Cooum River
            (13.0393, 80.2783),  # Buckingham Canal
            (13.0047, 80.2564),  # Pallikaranai Marsh
        ]
        
        # Calculate distance to nearest water body
        min_distance_to_water = min([
            np.sqrt((lat - wlat)**2 + (lon - wlon)**2) * 111  # Convert to km
            for wlat, wlon in water_bodies
        ])
        
        # Distance to coast (Chennai is coastal)
        coast_distance = abs(80.27 - lon) * 111
        
        # Risk decreases with distance from water (max effect within 5km)
        water_risk = max(0.1, 1 - (min(min_distance_to_water, coast_distance) / 5))
        
        return water_risk

    def is_flood_prone_area(self, lat, lon):
        """Check if location is in known flood-prone areas of Chennai"""
        flood_prone_areas = [
            (13.0827, 80.2707, 3),  # Central Chennai
            (13.0359, 80.2309, 2),  # Guindy
            (13.0067, 80.2206, 2),  # Velachery
            (13.0604, 80.2496, 2),  # Adyar
        ]
        
        for flood_lat, flood_lon, radius in flood_prone_areas:
            distance = np.sqrt((lat - flood_lat)**2 + (lon - flood_lon)**2) * 111
            if distance < radius:
                return True
        return False

    def train_ai_models(self, weather_df):
        print("Training AI prediction models...")
        
        # Simple training with available data
        features = ['temperature', 'humidity', 'population_density', 'green_cover']
        available_features = [col for col in features if col in weather_df.columns]
        
        if len(available_features) < 2:
            print("Not enough features available for training")
            return False
            
        X = weather_df[available_features].values
        y = weather_df.apply(lambda row: self.calculate_heat_index(row['temperature'], row['humidity']), axis=1)
        
        self.heat_index_model = RandomForestRegressor(n_estimators=20, random_state=42, max_depth=5)
        self.heat_index_model.fit(X, y)
        
        print("AI models trained successfully")
        return True

    def generate_predictions(self, spatial_df, weather_df):
        print("Generating hazard predictions for ALL locations...")
        
        predictions = []
        
        # Verify we have data for all cells
        spatial_cells = set(spatial_df['cell_id'])
        weather_cells = set(weather_df['cell_id'])
        missing_cells = spatial_cells - weather_cells
        
        if missing_cells:
            print(f"Warning: Missing weather data for {len(missing_cells)} cells")
        
        # Process EVERY spatial location
        for idx, location in spatial_df.iterrows():
            cell_id = location['cell_id']
            
            # Find weather data for this cell
            cell_weather = weather_df[weather_df['cell_id'] == cell_id]
            
            if len(cell_weather) > 0:
                weather = cell_weather.iloc[0]
                
                # Calculate heat risk
                try:
                    heat_features = [
                        weather['temperature'],
                        weather['humidity'],
                        location['population_density'],
                        location['green_cover']
                    ]
                    heat_index = self.heat_index_model.predict([heat_features])[0]
                except:
                    heat_index = self.calculate_heat_index(weather['temperature'], weather['humidity'])
                
                heat_risk = min(100, max(0, (heat_index - 30) * 3))
                
                # Calculate air quality risk (NEW improved method)
                aqi = self.calculate_air_quality_index(weather['pm25'])
                air_quality_risk = self.calculate_air_quality_risk(aqi)
                
                # Calculate flood risk (NEW improved method)
                flood_risk = self.calculate_flood_risk(location, weather)
                
                overall_risk = (heat_risk * 0.4 + air_quality_risk * 0.3 + flood_risk * 0.3)
                
            else:
                # No weather data - use safe defaults
                heat_index = 25.0
                heat_risk = 0.0
                aqi = 50.0
                air_quality_risk = 0.0
                flood_risk = 0.0
                overall_risk = 0.0
            
            predictions.append({
                'cell_id': cell_id,
                'latitude': location['latitude'],
                'longitude': location['longitude'],
                'heat_index': round(heat_index, 1),
                'heat_risk': round(heat_risk, 1),
                'air_quality_index': round(aqi, 1),
                'air_quality_risk': round(air_quality_risk, 1),
                'flood_risk': round(flood_risk, 1),
                'overall_risk': round(overall_risk, 1),
                'population_density': location['population_density'],
                'green_cover': location['green_cover']
            })
            
            # Progress indicator
            if (idx + 1) % 50 == 0:
                print(f"   Generated predictions for {idx + 1}/{len(spatial_df)} locations")
        
        df = pd.DataFrame(predictions)
        print(f"Generated {len(df)} predictions")
        return df

    def create_visualizations(self, predictions_df, weather_df):
        print("Generating analytical visualizations...")
        
        try:
            # Simple histogram of overall risk
            plt.figure(figsize=(10, 6))
            plt.hist(predictions_df['overall_risk'], bins=20, alpha=0.7, color='blue')
            plt.title('Overall Risk Distribution')
            plt.xlabel('Risk Level (%)')
            plt.ylabel('Number of Locations')
            plt.grid(True, alpha=0.3)
            plt.savefig('output/charts/risk_distribution.png', dpi=150, bbox_inches='tight')
            plt.close()
            
            # Scatter plot: population vs risk
            plt.figure(figsize=(12, 10))
            sc = plt.scatter(predictions_df['longitude'], predictions_df['latitude'], 
                        s=predictions_df['population_density']/100, 
                        c=predictions_df['overall_risk'], 
                        cmap='RdYlGn_r', alpha=0.6)
            plt.colorbar(sc, label='Overall Risk (%)')
            plt.xlabel('Longitude')
            plt.ylabel('Latitude')
            plt.title('Spatial Distribution of Risk and Population Density\n(Bubble size: Population Density)')
            plt.grid(True, alpha=0.3)
            plt.savefig('output/charts/population_vs_risk.png', dpi=150, bbox_inches='tight')
            plt.close()
            
            print("Visualizations generated")
        except Exception as e:
            print(f"Visualization error: {e}")

def main():
    predictor = HazardPredictor()
    
    try:
        print("Starting AI Digital Twin System...")
        
        # Clear previous outputs
        for file in ['spatial_data.csv', 'weather_data.csv', 'predictions.csv', 'city_hazards.geojson', 'summary_report.json']:
            if os.path.exists(f'output/{file}'):
                os.remove(f'output/{file}')
        
        # Step 1: Generate spatial data
        print("Step 1: Generating spatial data...")
        spatial_df = predictor.generate_spatial_data()
        
        # Step 2: Generate weather data
        print("Step 2: Generating weather data...")
        weather_df = predictor.simulate_weather_conditions(spatial_df)
        
        # Step 3: Train models
        print("Step 3: Training AI models...")
        success = predictor.train_ai_models(weather_df)
        if not success:
            print("Using fallback prediction methods")
        
        # Step 4: Generate predictions
        print("Step 4: Generating predictions...")
        predictions_df = predictor.generate_predictions(spatial_df, weather_df)
        
        # Step 5: Create visualizations
        print("Step 5: Creating visualizations...")
        predictor.create_visualizations(predictions_df, weather_df)
        
        # Save data files
        print("Step 6: Saving data files...")
        spatial_df.to_csv('output/spatial_data.csv', index=False)
        weather_df.to_csv('output/weather_data.csv', index=False)
        predictions_df.to_csv('output/predictions.csv', index=False)
        
        # Create GeoJSON
        print("Step 7: Creating GeoJSON...")
        geojson = {"type": "FeatureCollection", "features": []}
        for _, pred in predictions_df.iterrows():
            feature = {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [pred['longitude'], pred['latitude']]
                },
                "properties": {
                    "cell_id": pred['cell_id'],
                    "heat_risk": pred['heat_risk'],
                    "air_quality_risk": pred['air_quality_risk'],
                    "flood_risk": pred['flood_risk'],
                    "overall_risk": pred['overall_risk'],
                    "population_density": pred['population_density']
                }
            }
            geojson["features"].append(feature)
        
        with open('output/city_hazards.geojson', 'w') as f:
            json.dump(geojson, f, indent=2)
        
        # Create summary
        print("Step 8: Creating summary report...")
        summary = {
            "timestamp": datetime.now().isoformat(),
            "city": "Chennai",
            "total_locations": len(predictions_df),
            "risk_statistics": {
                "average_heat_risk": predictions_df['heat_risk'].mean(),
                "average_air_quality_risk": predictions_df['air_quality_risk'].mean(),
                "average_flood_risk": predictions_df['flood_risk'].mean(),
                "high_risk_locations": len(predictions_df[predictions_df['overall_risk'] > 70]),
            }
        }
        
        with open('output/summary_report.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        print("=" * 60)
        print("AI Digital Twin Successfully Generated!")
        print(f"Locations analyzed: {len(predictions_df)}")
        print(f"GeoJSON features: {len(geojson['features'])}")
        print(f"Average heat risk: {summary['risk_statistics']['average_heat_risk']:.1f}%")
        print(f"Average flood risk: {summary['risk_statistics']['average_flood_risk']:.1f}%")
        print(f"Average air quality risk: {summary['risk_statistics']['average_air_quality_risk']:.1f}%")
        print("=" * 60)
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()