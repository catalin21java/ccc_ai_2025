#!/usr/bin/env python3
"""
Solve Level 2: The Bird Love Score

Tasks:
1. Load data from level 2 (Vegetation, Insects, Urban Light, Bird Love Score)
2. Merge with temperature and humidity from level 1
3. Detect and convert temperatures mistakenly reported in °F to °C
4. Build a regression model to predict missing Bird Love Scores
5. Output predictions in CSV format
"""

import csv
import os
import glob
import math

# Number word parsing (reused from level 1)
NUMBER_WORDS = {
    'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
    'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10,
    'eleven': 11, 'twelve': 12, 'thirteen': 13, 'fourteen': 14,
    'fifteen': 15, 'sixteen': 16, 'seventeen': 17, 'eighteen': 18,
    'nineteen': 19, 'twenty': 20, 'thirty': 30, 'forty': 40,
    'fifty': 50, 'sixty': 60, 'seventy': 70, 'eighty': 80,
    'ninety': 90, 'hundred': 100
}

def parse_number_word(text):
    """Parse a number word (e.g., 'seventeen') to integer."""
    text = text.lower().strip()
    if text in NUMBER_WORDS:
        return NUMBER_WORDS[text]
    parts = text.split('-')
    if len(parts) == 2:
        tens = NUMBER_WORDS.get(parts[0], 0)
        ones = NUMBER_WORDS.get(parts[1], 0)
        if tens > 0 and ones > 0:
            return tens + ones
    return None

def parse_temperature(temp_str):
    """Parse temperature, handling numeric strings and number words."""
    try:
        return float(temp_str)
    except ValueError:
        num = parse_number_word(temp_str)
        if num is not None:
            return float(num)
        return None

def load_level1_data(level1_file):
    """Load temperature and humidity data from level 1."""
    temp_humidity = {}
    with open(level1_file, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        for row in reader:
            if len(row) >= 3:
                bop_id = int(row[0].strip())
                temp = parse_temperature(row[1].strip())
                humidity = float(row[2].strip())
                if temp is not None:
                    temp_humidity[bop_id] = {'temp': temp, 'humidity': humidity}
    return temp_humidity

def detect_fahrenheit_temps(temps):
    """Detect which temperatures are likely in Fahrenheit.
    Normal Celsius range: -50 to 50°C
    If temp > 50, likely Fahrenheit"""
    fahrenheit_indices = []
    for i, temp in enumerate(temps):
        if temp > 50:  # Likely Fahrenheit
            fahrenheit_indices.append(i)
    return fahrenheit_indices

def fahrenheit_to_celsius(f):
    """Convert Fahrenheit to Celsius."""
    return (f - 32) * 5 / 9

def load_level2_data(input_file, temp_humidity_data):
    """Load level 2 data and merge with temperature/humidity."""
    training_data = []
    missing_data = []
    
    with open(input_file, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        
        for row in reader:
            if len(row) < 5:
                continue
            
            bop_id = int(row[0].strip())
            vegetation = float(row[1].strip())
            insects = float(row[2].strip())
            urban_light = float(row[3].strip())
            bird_love_score = row[4].strip()
            
            # Get temperature and humidity from level 1
            temp = None
            humidity = None
            if bop_id in temp_humidity_data:
                temp = temp_humidity_data[bop_id]['temp']
                humidity = temp_humidity_data[bop_id]['humidity']
            
            data_point = {
                'bop_id': bop_id,
                'vegetation': vegetation,
                'insects': insects,
                'urban_light': urban_light,
                'temp': temp,
                'humidity': humidity
            }
            
            if bird_love_score == 'missing':
                missing_data.append(data_point)
            else:
                data_point['bird_love_score'] = float(bird_love_score)
                training_data.append(data_point)
    
    return training_data, missing_data

def prepare_features(data_list, include_polynomial=False):
    """Prepare feature matrix from data list."""
    features = []
    temps = []
    
    for dp in data_list:
        vegetation = dp['vegetation']
        insects = dp['insects']
        urban_light = dp['urban_light']
        
        feature_vec = [
            vegetation,
            insects,
            urban_light
        ]
        
        # Add polynomial features for better model capacity
        if include_polynomial:
            feature_vec.extend([
                vegetation * vegetation,
                insects * insects,
                urban_light * urban_light,
                vegetation * insects,
                vegetation * urban_light,
                insects * urban_light
            ])
        
        # Add temperature and humidity if available
        temp = dp.get('temp')
        humidity = dp.get('humidity')
        
        if temp is not None:
            feature_vec.append(temp)
            temps.append(temp)
            if include_polynomial:
                feature_vec.append(temp * temp)  # Temperature squared
        else:
            feature_vec.append(0)  # Missing value placeholder
            temps.append(None)
            if include_polynomial:
                feature_vec.append(0)
        
        if humidity is not None:
            feature_vec.append(humidity)
            if include_polynomial:
                feature_vec.append(humidity * humidity)  # Humidity squared
        else:
            feature_vec.append(0)  # Missing value placeholder
            if include_polynomial:
                feature_vec.append(0)
        
        features.append(feature_vec)
    
    return features, temps

def correct_temperatures(features, temps, has_polynomial=False):
    """Detect and correct Fahrenheit temperatures."""
    # Temperature column index depends on whether polynomial features are included
    if has_polynomial:
        temp_col_idx = 9  # After 3 base + 6 polynomial features
        temp_squared_idx = 10
    else:
        temp_col_idx = 3  # Temperature is 4th feature (index 3)
        temp_squared_idx = None
    
    for i in range(len(features)):
        if temps[i] is not None and temps[i] > 50:
            # Likely Fahrenheit, convert to Celsius
            f_temp = features[i][temp_col_idx]
            c_temp = fahrenheit_to_celsius(f_temp)
            features[i][temp_col_idx] = c_temp
            if temp_squared_idx is not None:
                features[i][temp_squared_idx] = c_temp * c_temp
            temps[i] = c_temp
    
    return features, temps

def mean(values):
    """Calculate mean."""
    return sum(values) / len(values) if values else 0

def std(values):
    """Calculate standard deviation."""
    if not values:
        return 1
    m = mean(values)
    variance = sum((x - m) ** 2 for x in values) / len(values)
    return math.sqrt(variance)

def normalize_features(features):
    """Normalize features using z-score."""
    if not features:
        return features
    
    num_features = len(features[0])
    normalized = []
    
    # Calculate mean and std for each feature
    feature_means = []
    feature_stds = []
    
    for col_idx in range(num_features):
        col_values = [row[col_idx] for row in features]
        feature_means.append(mean(col_values))
        feature_stds.append(std(col_values) if std(col_values) > 0 else 1)
    
    # Normalize
    for row in features:
        normalized_row = []
        for col_idx in range(num_features):
            if feature_stds[col_idx] > 0:
                normalized_row.append((row[col_idx] - feature_means[col_idx]) / feature_stds[col_idx])
            else:
                normalized_row.append(0)
        normalized.append(normalized_row)
    
    return normalized, feature_means, feature_stds

def denormalize_features(features, feature_means, feature_stds):
    """Denormalize features."""
    denormalized = []
    for row in features:
        denormalized_row = []
        for col_idx in range(len(row)):
            denormalized_row.append(row[col_idx] * feature_stds[col_idx] + feature_means[col_idx])
        denormalized.append(denormalized_row)
    return denormalized

class SimpleLinearRegression:
    """Simple linear regression implementation with L2 regularization."""
    
    def __init__(self, alpha=0.1):
        self.weights = None
        self.bias = 0
        self.alpha = alpha  # Regularization parameter
    
    def fit(self, X, y, learning_rate=0.01, iterations=1000):
        """Train using gradient descent with L2 regularization."""
        if not X or not y:
            return
        
        num_features = len(X[0])
        self.weights = [0.0] * num_features
        self.bias = 0.0
        
        n = len(X)
        
        for iteration in range(iterations):
            # Calculate predictions
            predictions = []
            for row in X:
                pred = self.bias
                for i in range(num_features):
                    pred += self.weights[i] * row[i]
                predictions.append(pred)
            
            # Calculate gradients with L2 regularization
            bias_gradient = sum(predictions[i] - y[i] for i in range(n)) / n
            
            weight_gradients = []
            for j in range(num_features):
                # Gradient includes regularization term
                grad = sum((predictions[i] - y[i]) * X[i][j] for i in range(n)) / n
                grad += self.alpha * self.weights[j]  # L2 regularization
                weight_gradients.append(grad)
            
            # Update parameters
            self.bias -= learning_rate * bias_gradient
            for j in range(num_features):
                self.weights[j] -= learning_rate * weight_gradients[j]
    
    def predict(self, X):
        """Make predictions."""
        predictions = []
        for row in X:
            pred = self.bias
            for i in range(len(self.weights)):
                pred += self.weights[i] * row[i]
            predictions.append(pred)
        return predictions

def solve_input_file(input_file, level1_file, fallback_training_file=None):
    """Solve a single input file."""
    # Load level 1 data
    temp_humidity_data = load_level1_data(level1_file)
    
    # Load level 2 data
    training_data, missing_data = load_level2_data(input_file, temp_humidity_data)
    
    # If no training data, use fallback training file
    if len(training_data) == 0 and fallback_training_file:
        print(f"  No training data in {input_file}, using {fallback_training_file} for training")
        training_data, _ = load_level2_data(fallback_training_file, temp_humidity_data)
    
    print(f"Training samples: {len(training_data)}")
    print(f"Missing samples: {len(missing_data)}")
    
    if len(training_data) == 0:
        raise ValueError(f"No training data available for {input_file}")
    
    # Use polynomial features only for larger datasets, simpler for smaller ones
    use_poly = len(training_data) > 500
    
    # Prepare features
    X_train, train_temps = prepare_features(training_data, include_polynomial=use_poly)
    X_missing, missing_temps = prepare_features(missing_data, include_polynomial=use_poly)
    
    # Extract target values
    y_train = [dp['bird_love_score'] for dp in training_data]
    
    # Correct Fahrenheit temperatures
    X_train, train_temps = correct_temperatures(X_train, train_temps, has_polynomial=use_poly)
    X_missing, missing_temps = correct_temperatures(X_missing, missing_temps, has_polynomial=use_poly)
    
    # Normalize features
    X_train_scaled, train_means, train_stds = normalize_features(X_train)
    
    # Normalize missing data using same parameters as training
    X_missing_scaled = []
    for row in X_missing:
        normalized_row = []
        for col_idx in range(len(row)):
            if train_stds[col_idx] > 0:
                normalized_row.append((row[col_idx] - train_means[col_idx]) / train_stds[col_idx])
            else:
                normalized_row.append(0)
        X_missing_scaled.append(normalized_row)
    
    # Optimize regularization - use specific values for level_2_b
    if len(training_data) > 1000:
        # For level_2_b, use low regularization to fit better
        alpha_val = 0.02
        iterations = 4000
        learning_rate = 0.008
    elif len(training_data) > 100:
        alpha_val = 0.1
        iterations = 3000
        learning_rate = 0.01
    else:
        alpha_val = 0.5
        iterations = 5000
        learning_rate = 0.01
    
    model = SimpleLinearRegression(alpha=alpha_val)
    model.fit(X_train_scaled, y_train, learning_rate=learning_rate, iterations=iterations)
    
    # Calculate training RMSE
    y_pred_train = model.predict(X_train_scaled)
    train_rmse = math.sqrt(sum((y_train[i] - y_pred_train[i]) ** 2 for i in range(len(y_train))) / len(y_train))
    print(f"  Training RMSE: {train_rmse:.4f}")
    
    # Predict missing values
    y_pred_missing = model.predict(X_missing_scaled)
    
    # Create predictions list
    predictions = []
    for dp, pred in zip(missing_data, y_pred_missing):
        predictions.append({
            'bop_id': dp['bop_id'],
            'bird_love_score': max(0, pred)  # Ensure non-negative
        })
    
    # Sort by BOP ID
    predictions.sort(key=lambda x: x['bop_id'])
    
    return predictions

def main():
    level1_file = 'all_data_from_level_1.in'
    input_files = sorted(glob.glob('level_2_*.in'))
    input_files = [f for f in input_files if 'sample' not in f]
    
    if not input_files:
        print("No input files found!")
        return
    
    # Use level_2_a as fallback training for level_2_c
    fallback_training = 'level_2_a.in' if 'level_2_a.in' in input_files else None
    
    for input_file in input_files:
        print(f"\n{'='*60}")
        print(f"Processing {input_file}")
        print(f"{'='*60}")
        
        # Use fallback training only for level_2_c
        fallback = fallback_training if 'level_2_c' in input_file else None
        predictions = solve_input_file(input_file, level1_file, fallback_training_file=fallback)
        
        # Write output
        output_file = input_file.replace('.in', '.out')
        with open(output_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['BOP', 'Bird Love Score [<3]'])
            for pred in predictions:
                writer.writerow([pred['bop_id'], f"{pred['bird_love_score']:.2f}"])
        
        print(f"\nOutput written to {output_file}")
        print(f"Predicted {len(predictions)} missing values")

if __name__ == '__main__':
    main()

