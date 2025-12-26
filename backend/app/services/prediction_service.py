"""Prediction Service - ML models for health risk prediction.

Currently implements breast cancer prediction using a RandomForest model
trained on a breast cancer dataset. If a local CSV dataset is available,
it will be used; otherwise we fall back to scikit-learn's built-in
"breast cancer" dataset.
"""
import os
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from app.models.analysis import RiskLevel, FeatureImportance, ModelType

logger = logging.getLogger(__name__)

# Model storage path
MODELS_DIR = Path(__file__).parent.parent / "ml" / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


class PredictionService:
    """Service for ML-based health risk predictions."""
    
    # Features expected for breast cancer model (Wisconsin dataset)
    BREAST_CANCER_FEATURES = [
        "radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean",
        "compactness_mean", "concavity_mean", "concave_points_mean", "symmetry_mean",
        "fractal_dimension_mean", "radius_se", "texture_se", "perimeter_se", "area_se",
        "smoothness_se", "compactness_se", "concavity_se", "concave_points_se",
        "symmetry_se", "fractal_dimension_se", "radius_worst", "texture_worst",
        "perimeter_worst", "area_worst", "smoothness_worst", "compactness_worst",
        "concavity_worst", "concave_points_worst", "symmetry_worst", "fractal_dimension_worst"
    ]
    
    def __init__(self):
        """Initialize prediction service and load/train models."""
        self.models: Dict[ModelType, Any] = {}
        self.scalers: Dict[ModelType, Any] = {}
        self._load_or_train_models()
    
    def _load_or_train_models(self):
        """Load existing models or train new ones."""
        bc_model_path = MODELS_DIR / "breast_cancer_model.joblib"
        bc_scaler_path = MODELS_DIR / "breast_cancer_scaler.joblib"
        
        if bc_model_path.exists() and bc_scaler_path.exists():
            logger.info("Loading existing breast cancer model...")
            self.models[ModelType.BREAST_CANCER] = joblib.load(bc_model_path)
            self.scalers[ModelType.BREAST_CANCER] = joblib.load(bc_scaler_path)
        else:
            logger.info("Training new breast cancer model...")
            self._train_breast_cancer_model()

    def _train_breast_cancer_model(self):
        """Train breast cancer prediction model.

        Preference order for training data:
        1. A local CSV dataset (e.g. "breast cancer.csv") if available.
        2. Fallback to scikit-learn's built-in breast cancer dataset.
        """
        X = None
        y = None

        # 1) Try to load user-provided CSV dataset
        csv_path: Optional[Path] = None
        csv_env = os.getenv("BREAST_CANCER_DATASET_PATH")
        if csv_env:
            csv_path = Path(csv_env)
        else:
            # Default: look for "breast cancer.csv" at the project root
            try:
                project_root = Path(__file__).resolve().parents[3]
            except IndexError:
                project_root = Path(__file__).resolve().parent
            csv_path = project_root / "breast cancer.csv"

        if csv_path and csv_path.exists():
            logger.info(f"Training breast cancer model from CSV dataset: {csv_path}")
            df = pd.read_csv(csv_path)
            # Drop unnamed columns that can appear due to trailing commas
            df = df.loc[:, ~df.columns.str.contains(r"^Unnamed", case=False)]

            # Normalise column names to match BREAST_CANCER_FEATURES
            rename_map = {
                "concave points_mean": "concave_points_mean",
                "concave points_se": "concave_points_se",
                "concave points_worst": "concave_points_worst",
            }
            df = df.rename(columns=rename_map)

            if "diagnosis" not in df.columns:
                raise ValueError("CSV dataset must contain a 'diagnosis' column")

            # Map diagnosis labels to numeric targets.
            # We keep the convention: 0 = malignant, 1 = benign.
            y = df["diagnosis"].map({"M": 0, "B": 1})
            if y.isnull().any():
                raise ValueError(
                    "Diagnosis column contains unexpected labels; expected only 'M' or 'B'"
                )

            missing_features = [
                f for f in self.BREAST_CANCER_FEATURES if f not in df.columns
            ]
            if missing_features:
                raise ValueError(
                    f"CSV dataset is missing required feature columns: {missing_features}"
                )

            X = df[self.BREAST_CANCER_FEATURES].to_numpy()
        else:
            # 2) Fallback to sklearn dataset if CSV is not available
            logger.warning(
                "CSV dataset not found; falling back to sklearn.load_breast_cancer()"
            )
            data = load_breast_cancer()
            X, y = data.data, data.target

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train Random Forest model
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X_train_scaled, y_train)

        # Evaluate
        accuracy = model.score(X_test_scaled, y_test)
        logger.info(f"Breast cancer model trained with accuracy: {accuracy:.4f}")

        # Save model and scaler
        joblib.dump(model, MODELS_DIR / "breast_cancer_model.joblib")
        joblib.dump(scaler, MODELS_DIR / "breast_cancer_scaler.joblib")

        self.models[ModelType.BREAST_CANCER] = model
        self.scalers[ModelType.BREAST_CANCER] = scaler
    
    def predict(
        self, 
        input_data: Dict[str, Any], 
        model_type: ModelType
    ) -> Tuple[float, RiskLevel, List[FeatureImportance]]:
        """
        Make a prediction using the specified model.
        
        Args:
            input_data: Dictionary of feature values
            model_type: Type of model to use
            
        Returns:
            Tuple of (risk_score, risk_level, feature_importances)
        """
        if model_type == ModelType.BREAST_CANCER:
            return self._predict_breast_cancer(input_data)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def _predict_breast_cancer(
        self, 
        input_data: Dict[str, Any]
    ) -> Tuple[float, RiskLevel, List[FeatureImportance]]:
        """Make breast cancer malignancy prediction."""
        model = self.models.get(ModelType.BREAST_CANCER)
        scaler = self.scalers.get(ModelType.BREAST_CANCER)
        
        if not model or not scaler:
            raise RuntimeError("Breast cancer model not loaded")
        
        # Prepare feature vector
        features = []
        for feat_name in self.BREAST_CANCER_FEATURES:
            value = input_data.get(feat_name, 0.0)
            features.append(float(value) if value else 0.0)

        X = np.array(features).reshape(1, -1)
        X_scaled = scaler.transform(X)

        # Get prediction probability
        # Note: In sklearn breast cancer dataset, 0 = malignant, 1 = benign
        probas = model.predict_proba(X_scaled)[0]
        malignant_prob = probas[0]  # Probability of malignant (class 0)
        
        # Determine risk level
        risk_level = self._get_risk_level(malignant_prob)
        
        # Get feature importances
        # Use both global feature importances and the current patient's scaled
        # feature values so that the "Facteurs les plus importants" section
        # is dynamic and specific to this prediction.
        importances = self._get_feature_importances(
            model,
            input_data,
            scaled_features=X_scaled[0],
        )
        
        return malignant_prob, risk_level, importances
    
    def _get_risk_level(self, score: float) -> RiskLevel:
        """Convert risk score to risk level category."""
        if score < 0.25:
            return RiskLevel.LOW
        elif score < 0.50:
            return RiskLevel.MODERATE
        elif score < 0.75:
            return RiskLevel.HIGH
        else:
            return RiskLevel.VERY_HIGH
    
    def _get_feature_importances(
        self,
        model,
        input_data: Dict[str, Any],
        scaled_features: Optional[np.ndarray] = None,
        top_n: int = 5,
    ) -> List[FeatureImportance]:
        """Get top N most important features for *this* prediction.

        We combine the model's global feature_importances_ with the current
        patient's (scaled) feature values so that the explanation is
        patient-specific instead of always showing the same global ranking.
        """

        global_importances = model.feature_importances_

        # Compute a per-feature score. When scaled_features are available
        # we weight global importance by the absolute scaled value so that
        # features which are both important *and* far from the mean for this
        # patient are highlighted.
        if scaled_features is not None:
            local_scores = np.abs(global_importances * scaled_features)
            total = float(local_scores.sum())
            if total > 0:
                scores = local_scores / total
            else:
                # Fallback to uniform scores if everything is zero
                scores = np.full_like(local_scores, 1.0 / len(local_scores))
        else:
            total = float(global_importances.sum())
            if total > 0:
                scores = global_importances / total
            else:
                scores = np.full_like(global_importances, 1.0 / len(global_importances))

        indices = np.argsort(scores)[::-1][:top_n]

        result: List[FeatureImportance] = []
        for idx in indices:
            feat_name = self.BREAST_CANCER_FEATURES[idx]

            # Sign of contribution based on the scaled value if available,
            # otherwise fall back to the raw input value.
            contrib_sign: str
            if scaled_features is not None:
                sf = float(scaled_features[idx])
                if sf > 0:
                    contrib_sign = "positive"
                elif sf < 0:
                    contrib_sign = "negative"
                else:
                    contrib_sign = "neutral"
            else:
                raw_val = input_data.get(feat_name, 0)
                try:
                    rv = float(raw_val)
                except (TypeError, ValueError):
                    rv = 0.0
                contrib_sign = "positive" if rv > 0 else "neutral"

            result.append(
                FeatureImportance(
                    feature_name=feat_name,
                    importance_score=float(scores[idx]),
                    contribution=contrib_sign,
                )
            )

        return result

