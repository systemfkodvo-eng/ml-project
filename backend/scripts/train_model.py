"""
Script to train/retrain ML models.
"""
import sys
sys.path.insert(0, '.')

from app.services.prediction_service import PredictionService


def main():
    """Train the breast cancer prediction model."""
    print("🧠 Training ML models...")
    print("=" * 50)
    
    # Initialize service (this triggers training if model doesn't exist)
    service = PredictionService()
    
    # Force retrain
    print("\n📊 Training Breast Cancer model...")
    service._train_breast_cancer_model()
    
    print("\n✅ Model training complete!")
    print("📁 Models saved to: backend/app/ml/models/")


if __name__ == "__main__":
    main()

