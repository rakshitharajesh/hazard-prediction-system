# advanced_models.py
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.cluster import DBSCAN
import joblib

class AdvancedHazardPredictor:
    def __init__(self):
        self.sequence_length = 24  # 24 hours of historical data
        
    def create_lstm_model(self):
        """Create LSTM model for time series prediction"""
        model = keras.Sequential([
            keras.layers.LSTM(64, return_sequences=True, 
                            input_shape=(self.sequence_length, 4)),
            keras.layers.Dropout(0.2),
            keras.layers.LSTM(32, return_sequences=False),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(16, activation='relu'),
            keras.layers.Dense(3, activation='sigmoid')  # 3 risks: heat, air, flood
        ])
        
        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def create_anomaly_detector(self):
        """Create autoencoder for anomaly detection"""
        encoder = keras.Sequential([
            keras.layers.Dense(32, activation='relu', input_shape=(10,)),
            keras.layers.Dense(16, activation='relu'),
            keras.layers.Dense(8, activation='relu')
        ])
        
        decoder = keras.Sequential([
            keras.layers.Dense(16, activation='relu', input_shape=(8,)),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dense(10, activation='sigmoid')
        ])
        
        autoencoder = keras.Sequential([encoder, decoder])
        autoencoder.compile(optimizer='adam', loss='mse')
        
        return autoencoder
    
    def spatial_clustering_analysis(self, spatial_data):
        """Perform spatial clustering of hazard zones"""
        coordinates = np.array([[loc['lat'], loc['lon']] for loc in spatial_data])
        
        # DBSCAN for spatial clustering
        clustering = DBSCAN(eps=0.02, min_samples=3).fit(coordinates)
        
        clusters = {}
        for i, label in enumerate(clustering.labels_):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(spatial_data[i])
        
        return clusters
    
    def ensemble_prediction(self, models, X):
        """Combine predictions from multiple models"""
        predictions = []
        for model in models:
            pred = model.predict(X)
            predictions.append(pred)
        
        # Weighted average based on model confidence
        ensemble_pred = np.mean(predictions, axis=0)
        return ensemble_pred

class RiskOptimizer:
    def __init__(self):
        self.resource_constraints = {
            'emergency_vehicles': 50,
            'medical_teams': 25,
            'evacuation_centers': 15
        }
    
    def optimize_resource_allocation(self, risk_predictions):
        """Optimize resource allocation based on risk predictions"""
        # Simple optimization algorithm
        allocations = {}
        
        for zone_id, risks in risk_predictions.items():
            total_risk = sum(risks.values())
            
            # Allocate resources proportional to risk
            allocations[zone_id] = {
                'emergency_vehicles': int(self.resource_constraints['emergency_vehicles'] * total_risk / 100),
                'medical_teams': int(self.resource_constraints['medical_teams'] * total_risk / 100),
                'evacuation_capacity': int(self.resource_constraints['evacuation_centers'] * total_risk / 100)
            }
        
        return allocations

# Train advanced models
def train_advanced_models():
    print("🤖 Training Advanced AI Models...")
    
    predictor = AdvancedHazardPredictor()
    
    # Generate synthetic training data
    X_train = np.random.random((1000, 24, 4))  # 1000 samples, 24 timesteps, 4 features
    y_train = np.random.random((1000, 3))     # 3 target risks
    
    # Train LSTM model
    lstm_model = predictor.create_lstm_model()
    lstm_model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)
    lstm_model.save('output/advanced_lstm_model.h5')
    
    # Train autoencoder
    autoencoder = predictor.create_anomaly_detector()
    X_ae = np.random.random((1000, 10))
    autoencoder.fit(X_ae, X_ae, epochs=20, batch_size=32, validation_split=0.2)
    autoencoder.save('output/anomaly_detector.h5')
    
    print("✅ Advanced models trained and saved")

if __name__ == "__main__":
    train_advanced_models()