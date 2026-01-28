import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os

class HotelPriceModel:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        
    def create_sample_data(self, num_samples=100):
        """Create sample data for testing"""
        np.random.seed(42)
        
        data = {
            'location_rating': np.random.randint(1, 6, num_samples),
            'hotel_star': np.random.randint(1, 6, num_samples),
            'room_type': np.random.randint(1, 4, num_samples),
            'season': np.random.randint(0, 2, num_samples),
            'num_guests': np.random.randint(1, 6, num_samples)
        } 
        
        # Create price based on features (with some randomness)
        base_price = 50
        data['price'] = (
            base_price +
            (data['location_rating'] * 20) +
            (data['hotel_star'] * 30) +
            (data['room_type'] * 40) +
            (data['season'] * 50) +
            (data['num_guests'] * 25) +
            np.random.normal(0, 20, num_samples)  # Add some noise
        )
        
        return pd.DataFrame(data)

    def train(self, data_path=None):
        """Train the model using either provided data or sample data"""
        try:
            if data_path and os.path.exists(data_path):
                print(f"Loading data from {data_path}")
                df = pd.read_csv(data_path)
            else:
                print("No data file provided or file not found. Using sample data...")
                df = self.create_sample_data()
                df.to_csv('sample_training_data.csv', index=False)
                print("Sample data saved to 'sample_training_data.csv'")

            # Validate columns
            required_columns = ['location_rating', 'hotel_star', 'room_type', 
                              'season', 'num_guests', 'price']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")

            # Prepare features and target
            X = df[required_columns[:-1]]  # All except price
            y = df['price']

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)

            # Train model
            self.model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            )
            
            self.model.fit(X_train_scaled, y_train)

            # Evaluate model
            train_score = self.model.score(X_train_scaled, y_train)
            test_score = self.model.score(X_test_scaled, y_test)

            print(f"\nModel Performance:")
            print(f"Training Score: {train_score:.4f}")
            print(f"Testing Score: {test_score:.4f}")

            # Save model and scaler
            joblib.dump(self.model, "hotel_price_model.pkl")
            joblib.dump(self.scaler, "hotel_price_scaler.pkl")
            print("\nModel and scaler saved successfully!")

            # Feature importance
            feature_importance = pd.DataFrame({
                'feature': required_columns[:-1],
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print("\nFeature Importance:")
            print(feature_importance)

            return True

        except Exception as e:
            print(f"Error during training: {str(e)}")
            return False

    def predict(self, features):
        """Make a prediction for given features"""
        try:
            if self.model is None:
                if os.path.exists("hotel_price_model.pkl"):
                    self.model = joblib.load("hotel_price_model.pkl")
                    self.scaler = joblib.load("hotel_price_scaler.pkl")
                else:
                    raise ValueError("Model not trained. Please train the model first.")

            # Validate input
            if len(features) != 5:
                raise ValueError("Expected 5 features: location_rating, hotel_star, room_type, season, num_guests")

            # Scale features
            features_scaled = self.scaler.transform([features])
            
            # Make prediction
            prediction = self.model.predict(features_scaled)[0]
            return prediction

        except Exception as e:
            print(f"Error during prediction: {str(e)}")
            return None

def main():
    # Create model instance
    model = HotelPriceModel()
    
    # Train model
    print("Training model...")
    model.train()  # Will use sample data if no file provided
    
    # Test prediction
    test_features = [
        5,    # location_rating (1-5)
        4,    # hotel_star (1-5)
        2,    # room_type (1=Standard, 2=Deluxe, 3=Suite)
        1,    # season (0=Off-season, 1=Peak season)
        2     # num_guests
    ]
    
    predicted_price = model.predict(test_features)
    if predicted_price is not None:
        print(f"\nTest Prediction:")
        print(f"Features: {test_features}")
        print(f"Predicted Price: ${predicted_price:.2f}")

if __name__ == "__main__":
    main()
    import time
print("\nPress Ctrl+C to exit or wait for 30 seconds...")
time.sleep(30)
