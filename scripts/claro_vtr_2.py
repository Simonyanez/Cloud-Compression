import numpy as np
import traceback
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
import pandas as pd

def solve_house_prices(training_data, test_data):
    """
    Predict office space prices using polynomial regression.
    
    Args:
        training_data (np.ndarray): Shape (N, F+1) where last column is price
        test_data (np.ndarray): Shape (T, F) features only
    
    Returns:
        np.ndarray: Predicted prices for test cases
    """
    
    # Step 1: Extract features and target from training data
    N, total_cols = training_data.shape
    F = total_cols - 1  # Number of features (last column is price)
    
    X_train = training_data[:, :F]      # First F columns are features
    y_train = training_data[:, F]       # Last column is price
    X_test = test_data                  # Test features
    
    print(f"DEBUG: Training data shape: {training_data.shape}")
    print(f"DEBUG: F={F} features, N={N} training samples")
    print(f"DEBUG: Test data shape: {test_data.shape}")
    print(f"DEBUG: X_train shape: {X_train.shape}")
    print(f"DEBUG: y_train shape: {y_train.shape}")
    
    # Display sample training data
    print(f"\nDEBUG: Sample training data:")
    for i in range(min(3, N)):
        print(f"  Sample {i+1}: features={X_train[i]}, price={y_train[i]:.2f}")
    
    # Display sample test data
    print(f"\nDEBUG: Sample test data:")
    for i in range(min(3, len(X_test))):
        print(f"  Test {i+1}: features={X_test[i]}")
    
    # Step 2: Feature analysis
    print(f"\nDEBUG: Feature statistics:")
    print(f"  Feature means: {np.mean(X_train, axis=0)}")
    print(f"  Feature stds: {np.std(X_train, axis=0)}")
    print(f"  Target mean: {np.mean(y_train):.2f}")
    print(f"  Target std: {np.std(y_train):.2f}")
    
    # Step 3: Model selection with cross-validation
    best_model = None
    best_score = -np.inf
    best_degree = 1
    
    print(f"\nDEBUG: Testing different polynomial degrees...")
    
    for degree in range(1, 4):  # Try degrees 1, 2, 3
        print(f"\nDEBUG: Evaluating polynomial degree {degree}")
        
        # Create pipeline with standardization + polynomial features + linear regression
        model = Pipeline([
            ('scaler', StandardScaler()),                    # Standardize features
            ('poly', PolynomialFeatures(degree=degree, 
                                      include_bias=True)),   # Generate polynomial features
            ('linear', LinearRegression())                   # Linear regression
        ])
        
        # Perform cross-validation to evaluate model
        try:
            cv_scores = cross_val_score(model, X_train, y_train, 
                                      cv=min(5, N//2), 
                                      scoring='r2',
                                      n_jobs=-1)
            
            mean_score = np.mean(cv_scores)
            std_score = np.std(cv_scores)
            
            print(f"  CV R² scores: {cv_scores}")
            print(f"  Mean CV R²: {mean_score:.4f} (+/- {std_score:.4f})")
            
            # Select best model based on CV score
            if mean_score > best_score:
                best_score = mean_score
                best_model = model
                best_degree = degree
                
        except Exception as e:
            print(f"  Error with degree {degree}: {e}")
            continue
    
    print(f"\nDEBUG: Selected polynomial degree: {best_degree}")
    print(f"DEBUG: Best CV R² score: {best_score:.4f}")
    
    # Step 4: Train final model on all training data
    print(f"\nDEBUG: Training final model...")
    best_model.fit(X_train, y_train)
    
    # Evaluate training performance
    train_score = best_model.score(X_train, y_train)
    print(f"DEBUG: Training R² score: {train_score:.4f}")
    
    # Step 5: Make predictions
    print(f"\nDEBUG: Making predictions...")
    predictions = best_model.predict(X_test)
    
    # Display predictions
    print(f"\nDEBUG: Predictions:")
    for i, price in enumerate(predictions):
        print(f"  Test case {i+1}: ${price:.2f}")
    
    print(f"\nDEBUG: Prediction statistics:")
    print(f"  Mean predicted price: ${np.mean(predictions):.2f}")
    print(f"  Std predicted price: ${np.std(predictions):.2f}")
    print(f"  Min predicted price: ${np.min(predictions):.2f}")
    print(f"  Max predicted price: ${np.max(predictions):.2f}")
    
    return predictions

def create_sample_data():
    """
    Create sample data for testing (simulating pandas DataFrame.values)
    """
    # Sample training data: 2 features + price
    np.random.seed(42)
    
    # Generate synthetic data where price is polynomial function of features
    n_samples = 20
    F = 2
    
    # Features between 0 and 1
    features = np.random.random((n_samples, F))
    
    # Price as polynomial function: price = 100 + 50*x1 + 30*x2 + 20*x1^2 + 10*x1*x2
    prices = (100 + 
             50 * features[:, 0] + 
             30 * features[:, 1] + 
             20 * features[:, 0]**2 + 
             10 * features[:, 0] * features[:, 1] +
             np.random.normal(0, 5, n_samples))  # Add noise
    
    # Combine features and prices
    training_data = np.column_stack([features, prices])
    
    # Generate test data (features only)
    test_features = np.random.random((5, F))
    
    return training_data, test_features

def create_test_case_data():
    """
    Create the exact test case data provided in the problem.
    
    Sample Input: F=2, N=100, T=4
    Expected Output: [180.38, 1312.07, 440.13, 343.72]
    """
    print("DEBUG: Creating exact test case data...")
    
    # Training data: 100 samples with 2 features + price
    training_data_raw = [
        [0.44, 0.68, 511.14], [0.99, 0.23, 717.1], [0.84, 0.29, 607.91], [0.28, 0.45, 270.4], [0.07, 0.83, 289.88],
        [0.66, 0.8, 830.85], [0.73, 0.92, 1038.09], [0.57, 0.43, 455.19], [0.43, 0.89, 640.17], [0.27, 0.95, 511.06],
        [0.43, 0.06, 177.03], [0.87, 0.91, 1242.52], [0.78, 0.69, 891.37], [0.9, 0.94, 1339.72], [0.41, 0.06, 169.88],
        [0.52, 0.17, 276.05], [0.47, 0.66, 517.43], [0.65, 0.43, 522.25], [0.85, 0.64, 932.21], [0.93, 0.44, 851.25],
        [0.41, 0.93, 640.11], [0.36, 0.43, 308.68], [0.78, 0.85, 1046.05], [0.69, 0.07, 332.4], [0.04, 0.52, 171.85],
        [0.17, 0.15, 109.55], [0.68, 0.13, 361.97], [0.84, 0.6, 872.21], [0.38, 0.4, 303.7], [0.12, 0.65, 256.38],
        [0.62, 0.17, 341.2], [0.79, 0.97, 1194.63], [0.82, 0.04, 408.6], [0.91, 0.53, 895.54], [0.35, 0.85, 518.25],
        [0.57, 0.69, 638.75], [0.52, 0.22, 301.9], [0.31, 0.15, 163.38], [0.6, 0.02, 240.77], [0.99, 0.91, 1449.05],
        [0.48, 0.76, 609.0], [0.3, 0.19, 174.59], [0.58, 0.62, 593.45], [0.65, 0.17, 355.96], [0.6, 0.69, 671.46],
        [0.95, 0.76, 1193.7], [0.47, 0.23, 278.88], [0.15, 0.96, 411.4], [0.01, 0.03, 42.08], [0.26, 0.23, 166.19],
        [0.01, 0.11, 58.62], [0.45, 0.87, 642.45], [0.09, 0.97, 368.14], [0.96, 0.25, 702.78], [0.63, 0.58, 615.74],
        [0.06, 0.42, 143.79], [0.1, 0.24, 109.0], [0.26, 0.62, 328.28], [0.41, 0.15, 205.16], [0.91, 0.95, 1360.49],
        [0.83, 0.64, 905.83], [0.44, 0.64, 487.33], [0.2, 0.4, 202.76], [0.43, 0.12, 202.01], [0.21, 0.22, 148.87],
        [0.88, 0.4, 745.3], [0.31, 0.87, 503.04], [0.99, 0.99, 1563.82], [0.23, 0.26, 165.21], [0.79, 0.12, 438.4],
        [0.02, 0.28, 98.47], [0.89, 0.48, 819.63], [0.02, 0.56, 174.44], [0.92, 0.03, 483.13], [0.72, 0.34, 534.24],
        [0.3, 0.99, 572.31], [0.86, 0.66, 957.61], [0.47, 0.65, 518.29], [0.79, 0.94, 1143.49], [0.82, 0.96, 1211.31],
        [0.9, 0.42, 784.74], [0.19, 0.62, 283.7], [0.7, 0.57, 684.38], [0.7, 0.61, 719.46], [0.69, 0.0, 292.23],
        [0.98, 0.3, 775.68], [0.3, 0.08, 130.77], [0.85, 0.49, 801.6], [0.73, 0.01, 323.55], [1.0, 0.23, 726.9],
        [0.42, 0.94, 661.12], [0.49, 0.98, 771.11], [0.89, 0.68, 1016.14], [0.22, 0.46, 237.69], [0.34, 0.5, 325.89],
        [0.99, 0.13, 636.22], [0.28, 0.46, 272.12], [0.87, 0.36, 696.65], [0.23, 0.87, 434.53], [0.77, 0.36, 593.86]
    ]
    
    # Test data: 4 samples with 2 features each
    test_data_raw = [
        [0.05, 0.54],
        [0.91, 0.91], 
        [0.31, 0.76],
        [0.51, 0.31]
    ]
    
    # Expected outputs for validation
    expected_outputs = [180.38, 1312.07, 440.13, 343.72]
    
    # Convert to numpy arrays
    training_data = np.array(training_data_raw)
    test_data = np.array(test_data_raw)
    
    print(f"DEBUG: Test case data created:")
    print(f"  Training data shape: {training_data.shape} (F=2, N=100)")
    print(f"  Test data shape: {test_data.shape} (T=4)")
    print(f"  Expected outputs: {expected_outputs}")
    
    # Verify constraints
    F = 2
    N = 100
    T = 4
    
    X_train = training_data[:, :F]
    y_train = training_data[:, F]
    X_test = test_data
    
    print(f"DEBUG: Constraint verification:")
    print(f"  F={F} (1 <= F <= 5): {'✓' if 1 <= F <= 5 else '✗'}")
    print(f"  N={N} (5 <= N <= 100): {'✓' if 5 <= N <= 100 else '✗'}")
    print(f"  T={T} (1 <= T <= 100): {'✓' if 1 <= T <= 100 else '✗'}")
    print(f"  Feature ranges: [{np.min(X_train):.3f}, {np.max(X_train):.3f}] and [{np.min(X_test):.3f}, {np.max(X_test):.3f}]")
    print(f"  Price range: [{np.min(y_train):.2f}, {np.max(y_train):.2f}]")
    print(f"  Features in [0,1]: {'✓' if np.all(X_train >= 0) and np.all(X_train <= 1) and np.all(X_test >= 0) and np.all(X_test <= 1) else '✗'}")
    print(f"  Prices in [0,1M]: {'✓' if np.all(y_train >= 0) and np.all(y_train <= 1e6) else '✗'}")
    
    return training_data, test_data, expected_outputs

if __name__ == "__main__":
    print("="*60)
    print("HOUSE PRICES PREDICTION - STANDARDIZED INPUT VERSION")
    print("="*60)
    
    # Test with the exact test case from the problem
    print("\n" + "="*40)
    print("TESTING WITH EXACT TEST CASE")
    print("="*40)
    
    training_data, test_data, expected_outputs = create_test_case_data()
    
    try:
        print("\n" + "-"*30)
        print("BASIC POLYNOMIAL REGRESSION")
        print("-"*30)
        
        predictions_basic = solve_house_prices(training_data, test_data)
        
        print(f"\nRESULTS COMPARISON:")
        print(f"{'Test Case':<10} {'Expected':<12} {'Predicted':<12} {'Difference':<12}")
        print("-" * 50)
        for i, (expected, predicted) in enumerate(zip(expected_outputs, predictions_basic)):
            diff = abs(expected - predicted)
            print(f"Test {i+1:<5} {expected:<12.2f} {predicted:<12.2f} {diff:<12.2f}")
        
        # Calculate overall accuracy metrics
        mse = np.mean((np.array(expected_outputs) - predictions_basic)**2)
        mae = np.mean(np.abs(np.array(expected_outputs) - predictions_basic))
        print(f"\nAccuracy Metrics:")
        print(f"Mean Squared Error (MSE): {mse:.2f}")
        print(f"Mean Absolute Error (MAE): {mae:.2f}")
    except Exception as e:
        print(f"Execution failed with error {e} \n {traceback.format_exc()}")
