"""MLflow pyfunc wrapper served by the Azure ML managed online endpoint: returns the fraud probability and the
decline decision at the threshold chosen on the validation period (not just a 0/1 class)."""
from __future__ import annotations

import json

import mlflow.pyfunc
import pandas as pd


class FraudScorer(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        import xgboost as xgb
        self.xgb = xgb
        self.booster = xgb.Booster()
        self.booster.load_model(context.artifacts["xgb_model"])
        meta = json.loads(open(context.artifacts["meta"]).read())
        self.features, self.threshold = meta["features"], float(meta["threshold"])

    def predict(self, context, model_input: pd.DataFrame, params=None) -> pd.DataFrame:
        X = model_input[self.features].astype(float)
        p = self.booster.predict(self.xgb.DMatrix(X.to_numpy(), feature_names=self.features))
        return pd.DataFrame({"fraud_probability": p.round(6), "decline": (p >= self.threshold).astype(int)})
