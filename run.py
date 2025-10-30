#!/usr/bin/env python3
"""
AI-Powered Urban Hazard Digital Twin - Main Script (No Emojis)
"""

import json
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
import joblib
import os

print("Initializing AI-Powered Urban Hazard Digital Twin...")
print("Focus Area: Chennai Metropolitan Area")
print("Time Period: Real-time + 48-hour forecast")
print("=" * 60)

# Create output directory
os.makedirs("output/charts", exist_ok=True)

class HazardPredictor:
    def __init__(self):
        self.chennai_center = (13.0827, 80.2707)
        self.grid_size = 10
        
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
                
                dist_center = np.sqrt((lat-13.0827)*2 + (lon-80.2707)*2)
                urban_heat_bias = max(0, 2 - dist_center * 10)
                
                spatial_data.append({
                    'cell_id': f"{i}_{j}",
                    'latitude': lat,
                    'longitude': lon,
                    'population_density': max(1000, population_density),
                    'building_height': max(5, building_height),
                    'green_cover': green_cover,
                    'urban_heat_bias': urban_heat_bias,
                    'elevation': max(0, 10 - dist_center * 100)
                })
        
        return pd.DataFrame(spatial_data)
    
    def simulate_weather_conditions(self, spatial_df):
        print("Simulating meteorological conditions...")
        
        current_time = datetime.now()
        time_steps = pd.date_range(current_time, periods=48, freq='H')
        
        weather_data = []
        for t, time in enumerate(time_steps):
            hour = time.hour
            base_temp = 28 + 8 * np.sin(2 * np.pi * (hour - 14) / 24)
            
            for _, location in spatial_df.iterrows():
                uhi_effect = location['urban_heat_bias'] * (1 + 0.2 * np.sin(2 * np.pi * t/24))
                green_cooling = location['green_cover'] * 0.1
                height_effect = location['building_height'] * 0.02
                
                temperature = base_temp + uhi_effect - green_cooling + height_effect
                humidity = 70 - (temperature - 25) * 2 + np.random.normal(0, 5)
                
                rain_prob = 0.3 if 6 <= hour <= 18 else 0.1
                rainfall = np.random.exponential(2) if np.random.random() < rain_prob else 0
                
                traffic_factor = min(1.0, location['population_density'] / 20000)
                industrial_pm = np.random.normal(20, 10) * traffic_factor
                background_pm = np.random.normal(15, 5)
                pm25 = max(5, industrial_pm + background_pm - rainfall * 0.5)
                
                weather_data.append({
                    'timestamp': time.isoformat(),
                    'cell_id': location['cell_id'],
                    'temperature': round(temperature, 1),
                    'humidity': max(10, min(100, humidity)),
                    'rainfall': round(rainfall, 1),
                    'pm25': round(pm25, 1),
                    'wind_speed': np.random.weibull(2) * 10,
                    'population_density': location['population_density'],
                    'green_cover': location['green_cover'],
                    'building_height': location['building_height']
                })
        
        return pd.DataFrame(weather_data)
    
    def calculate_heat_index(self, temp, humidity):
        hi = temp + 0.5 * (temp - 20) * (humidity - 50) / 50
        return max(temp, hi)
    
    def predict_flood_risk(self, rainfall, antecedent_rain, drainage_capacity):
        api = antecedent_rain * 0.8 + rainfall
        drainage_efficiency = max(0.1, 1 - drainage_capacity / 100000)
        flood_risk = min(100, api * drainage_efficiency * 10)
        return flood_risk
    
    def train_ai_models(self, weather_df):
        print("Training AI prediction models...")
        
        features = ['temperature', 'humidity', 'population_density', 'green_cover']
        available_features = [col for col in features if col in weather_df.columns]
        print(f"Using features: {available_features}")
        
        if len(available_features) < 2:
            print("Not enough features available for training")
            return False
            
        X = weather_df[available_features].values
        y = weather_df.apply(lambda row: self.calculate_heat_index(row['temperature'], row['humidity']), axis=1)
        
        self.heat_index_model = RandomForestRegressor(n_estimators=50, random_state=42, max_depth=5)
        self.heat_index_model.fit(X, y)
        
        aqi_features = ['pm25', 'temperature', 'wind_speed', 'population_density']
        available_aqi_features = [col for col in aqi_features if col in weather_df.columns]
        
        if len(available_aqi_features) >= 2:
            X_aqi = weather_df[available_aqi_features].values
            y_aqi = weather_df['pm25'] * 2
            self.aqi_model = RandomForestRegressor(n_estimators=50, random_state=42, max_depth=5)
            self.aqi_model.fit(X_aqi, y_aqi)
        else:
            print("Not enough features for AQI model, using simple calculation")
            self.aqi_model = None
        
        joblib.dump(self.heat_index_model, 'output/heat_index_model.pkl')
        if self.aqi_model:
            joblib.dump(self.aqi_model, 'output/aqi_model.pkl')
        
        print("AI models trained and saved")
        return True
    
    def generate_predictions(self, spatial_df, weather_df):
        print("Generating hazard predictions...")
        
        predictions = []
        current_data = weather_df[weather_df['timestamp'] == weather_df['timestamp'].max()]
        
        for _, location in spatial_df.iterrows():
            cell_id = location['cell_id']
            loc_weather = current_data[current_data['cell_id'] == cell_id]
            
            if len(loc_weather) == 0:
                continue
                
            loc_weather = loc_weather.iloc[0]
            
            heat_features = [
                loc_weather['temperature'],
                loc_weather['humidity'],
                location['population_density'],
                location['green_cover']
            ]
            
            try:
                heat_index = self.heat_index_model.predict([heat_features])[0]
                heat_risk = min(100, max(0, (heat_index - 30) * 3))
            except:
                heat_index = self.calculate_heat_index(loc_weather['temperature'], loc_weather['humidity'])
                heat_risk = min(100, max(0, (heat_index - 30) * 3))
            
            if self.aqi_model:
                aqi_features = [
                    loc_weather['pm25'],
                    loc_weather['temperature'],
                    loc_weather['wind_speed'],
                    location['population_density']
                ]
                aqi = self.aqi_model.predict([aqi_features])[0]
            else:
                aqi = loc_weather['pm25'] * 2
                
            air_quality_risk = min(100, max(0, (aqi - 20) * 2))
            
            antecedent_rain = weather_df[(weather_df['cell_id'] == cell_id)]['rainfall'].mean()
            if pd.isna(antecedent_rain):
                antecedent_rain = 0
                
            flood_risk = self.predict_flood_risk(
                loc_weather['rainfall'],
                antecedent_rain,
                location['population_density']
            )
            
            overall_risk = (heat_risk * 0.4 + air_quality_risk * 0.3 + flood_risk * 0.3)
            
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
        
        return pd.DataFrame(predictions)
    
    def create_visualizations(self, predictions_df, weather_df):
        print("Generating analytical visualizations...")
        
        plt.style.use('default')
        
        if len(predictions_df) == 0:
            print("No predictions to visualize")
            return
            
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            axes[0,0].hist(predictions_df['heat_risk'], bins=10, alpha=0.7, color='red')
            axes[0,0].set_title('Heat Risk Distribution', fontsize=14, fontweight='bold')
            axes[0,0].set_xlabel('Heat Risk Index')
            axes[0,0].set_ylabel('Number of Locations')
            axes[0,0].grid(True, alpha=0.3)
            
            if 'population_density' in predictions_df.columns and 'air_quality_index' in predictions_df.columns:
                axes[0,1].scatter(predictions_df['population_density'], 
                                 predictions_df['air_quality_index'], 
                                 alpha=0.6, color='blue')
                axes[0,1].set_title('Air Quality vs Population Density', fontsize=14, fontweight='bold')
                axes[0,1].set_xlabel('Population Density')
                axes[0,1].set_ylabel('Air Quality Index')
                axes[0,1].grid(True, alpha=0.3)
            
            flood_risk_sorted = predictions_df.nlargest(10, 'flood_risk')
            axes[1,0].bar(range(len(flood_risk_sorted)), flood_risk_sorted['flood_risk'], 
                         color='blue', alpha=0.7)
            axes[1,0].set_title('Top Flood Risk Locations', fontsize=14, fontweight='bold')
            axes[1,0].set_xlabel('Location Index')
            axes[1,0].set_ylabel('Flood Risk Index')
            
            risk_columns = ['heat_risk', 'air_quality_risk', 'flood_risk']
            available_risk_columns = [col for col in risk_columns if col in predictions_df.columns]
            
            if len(available_risk_columns) >= 2:
                risk_corr = predictions_df[available_risk_columns].corr()
                im = axes[1,1].imshow(risk_corr, cmap='coolwarm', vmin=-1, vmax=1)
                axes[1,1].set_xticks(range(len(available_risk_columns)))
                axes[1,1].set_yticks(range(len(available_risk_columns)))
                axes[1,1].set_xticklabels(available_risk_columns, rotation=45)
                axes[1,1].set_yticklabels(available_risk_columns)
                axes[1,1].set_title('Risk Correlation Matrix', fontsize=14, fontweight='bold')
                
                for i in range(len(available_risk_columns)):
                    for j in range(len(available_risk_columns)):
                        axes[1,1].text(j, i, f'{risk_corr.iloc[i, j]:.2f}', 
                                      ha='center', va='center', fontweight='bold')
            
            plt.tight_layout()
            plt.savefig('output/charts/risk_analysis.png', dpi=150, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"Error creating visualization: {e}")
        
        try:
            plt.figure(figsize=(12, 8))
            sample_locations = predictions_df['cell_id'].unique()[:3]
            
            for i, loc in enumerate(sample_locations):
                loc_data = weather_df[weather_df['cell_id'] == loc].tail(24)
                if len(loc_data) > 0:
                    times = range(len(loc_data))
                    
                    plt.subplot(2, 1, 1)
                    plt.plot(times, loc_data['temperature'], marker='o', label=f'Location {i+1}')
                    plt.ylabel('Temperature (°C)')
                    plt.title('24-hour Temperature Trend', fontweight='bold')
                    plt.legend()
                    plt.grid(True, alpha=0.3)
                    
                    plt.subplot(2, 1, 2)
                    plt.plot(times, loc_data['pm25'], marker='s', label=f'Location {i+1}')
                    plt.xlabel('Time Steps')
                    plt.ylabel('PM2.5 (µg/m³)')
                    plt.title('24-hour Air Quality Trend', fontweight='bold')
                    plt.legend()
                    plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('output/charts/time_series_analysis.png', dpi=150, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"Error creating time series visualization: {e}")
        
        print("Visualizations generated")

def main():
    predictor = HazardPredictor()
    
    try:
        spatial_df = predictor.generate_spatial_data()
        weather_df = predictor.simulate_weather_conditions(spatial_df)
        
        success = predictor.train_ai_models(weather_df)
        if not success:
            print("Using fallback prediction methods")
        
        predictions_df = predictor.generate_predictions(spatial_df, weather_df)
        predictor.create_visualizations(predictions_df, weather_df)
        
        spatial_df.to_csv('output/spatial_data.csv', index=False)
        weather_df.to_csv('output/weather_data.csv', index=False)
        predictions_df.to_csv('output/predictions.csv', index=False)
        
        geojson = {"type": "FeatureCollection", "features": []}
        for _, pred in predictions_df.iterrows():
            feature = {
                "type": "Feature",
                "geometry": {"type": "Point","coordinates": [pred['longitude'], pred['latitude']]},
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
        
        summary = {
            "timestamp": datetime.now().isoformat(),
            "city": "Chennai",
            "total_locations": len(predictions_df),
            "risk_statistics": {
                "average_heat_risk": predictions_df['heat_risk'].mean() if len(predictions_df) > 0 else 0,
                "average_air_quality_risk": predictions_df['air_quality_risk'].mean() if len(predictions_df) > 0 else 0,
                "average_flood_risk": predictions_df['flood_risk'].mean() if len(predictions_df) > 0 else 0,
                "high_risk_locations": len(predictions_df[predictions_df['overall_risk'] > 70]) if len(predictions_df) > 0 else 0,
            },
            "model_performance": {
                "status": "Operational",
                "locations_processed": len(predictions_df)
            }
        }
        
        with open('output/summary_report.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        print("=" * 60)
        print("AI Digital Twin Successfully Generated!")
        print(f"Output files created in /output directory")
        print(f"Locations analyzed: {len(predictions_df)}")
        if len(predictions_df) > 0:
            print(f"Average heat risk: {summary['risk_statistics']['average_heat_risk']:.1f}%")
            print(f"Average flood risk: {summary['risk_statistics']['average_flood_risk']:.1f}%")
        print("=" * 60)
        
    except Exception as e:
        print(f"Error in main execution: {e}")
        print("Try running with simpler configuration...")

if __name__ == "__main__":
    main()
