"""
Machine Learning Random Forest Sector Rotator for dynamic macro & momentum sector allocation.
"""

from typing import Dict, List, Optional, Tuple

import pandas as pd
from sklearn.ensemble import RandomForestClassifier


class SectorRotatorML:
    """
    Random Forest Machine Learning model for predicting leading market sectors
    based on macro signals (Treasury yields, inflation) and sector momentum features.
    """

    def __init__(self, n_estimators: int = 100, random_state: int = 42):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators, random_state=random_state
        )
        self.is_fitted = False
        self.feature_names: List[str] = []

    def build_features(
        self,
        sector_returns: pd.DataFrame,
        macro_series: Optional[pd.DataFrame] = None,
        lookback_windows: List[int] = [21, 63, 126, 252],
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Engineers momentum, volatility, and macro features for sector prediction.
        Target label (y): Sector index with the highest 1-month forward return.
        """
        features_list = []

        for window in lookback_windows:
            mom = sector_returns.rolling(window).mean()
            vol = sector_returns.rolling(window).std()
            mom.columns = [f"{col}_mom_{window}" for col in sector_returns.columns]
            vol.columns = [f"{col}_vol_{window}" for col in sector_returns.columns]
            features_list.extend([mom, vol])

        if macro_series is not None:
            features_list.append(macro_series)

        X = pd.concat(features_list, axis=1).dropna()

        # Forward 21-day returns to determine top sector
        fwd_returns = sector_returns.shift(-21).loc[X.index]
        y = fwd_returns.idxmax(axis=1)

        valid_mask = y.notna()
        X = X.loc[valid_mask]
        y = y.loc[valid_mask]

        self.feature_names = list(X.columns)
        return X, y

    def train(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """
        Trains the Random Forest sector rotator classifier.
        """
        self.model.fit(X, y)
        self.is_fitted = True
        accuracy = float(self.model.score(X, y))
        return {"training_accuracy": accuracy}

    def predict_sector_weights(
        self, current_features: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Predicts target sector weights based on class probabilities from the Random Forest model.
        """
        if not self.is_fitted:
            # Equal weighting fallback
            sectors = [
                c.replace("_mom_21", "") for c in self.feature_names if "_mom_21" in c
            ]
            if not sectors:
                return {}
            w = 1.0 / len(sectors)
            return {s: w for s in sectors}

        probs = self.model.predict_proba(current_features.iloc[[-1]])[0]
        classes = self.model.classes_

        weight_dict = {cls: float(p) for cls, p in zip(classes, probs)}
        return weight_dict

    def get_feature_importances(self) -> Dict[str, float]:
        """
        Returns feature importance scores from the trained model.
        """
        if not self.is_fitted:
            return {}
        importances = self.model.feature_importances_
        return {name: float(imp) for name, imp in zip(self.feature_names, importances)}
