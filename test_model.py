#!/usr/bin/env python3
"""
Test script to verify the model can be loaded properly
"""
import pickle
import os
import sys

def test_model_loading():
    """Test if the model can be loaded successfully"""
    try:
        # Try to load the model
        model_path = 'Diabetes-Prediction-ML-Model.sav'
        
        if not os.path.exists(model_path):
            print(f"❌ Model file not found: {model_path}")
            return False
            
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
            
        print("✅ Model loaded successfully!")
        print(f"Model type: {type(model)}")
        
        # Test a simple prediction
        import numpy as np
        test_data = np.array([[1, 85, 66, 29, 0, 26.6, 0.351, 31]])
        prediction = model.predict(test_data)
        print(f"✅ Test prediction successful: {prediction}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading model: {str(e)}")
        return False

if __name__ == "__main__":
    success = test_model_loading()
    sys.exit(0 if success else 1) 