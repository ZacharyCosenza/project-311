import pickle

import mlflow.pyfunc
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from modeling.pipelines.features.nodes import group_feature_cols


class GroupedCallModel:
    """The per-group XGBRegressors as one object, versioned and loaded as a unit.

    Previously these travelled as a bare ``dict[str, XGBRegressor]``, which left the
    feature-column convention and the "sum every group into one total" rule restated at
    each call site (rank_districts, compute_grouped_metrics, the four plotting nodes).
    Holding shared_feature_cols and max_lag_weeks alongside the estimators means
    feature_cols() has one definition that cannot drift from the one training used.

    Iteration and indexing are preserved so callers that legitimately want one head at a
    time (per-group metrics, per-group SHAP) read the same as they did against the dict.
    """

    def __init__(
        self, models: dict[str, XGBRegressor], shared_feature_cols: list, max_lag_weeks: int,
    ) -> None:
        self.models = models
        self.shared_feature_cols = list(shared_feature_cols)
        self.max_lag_weeks = max_lag_weeks

    @property
    def groups(self) -> list[str]:
        return list(self.models)

    def feature_cols(self, group: str) -> list:
        return group_feature_cols(group, self.shared_feature_cols, self.max_lag_weeks)

    def predict_group(self, df: pd.DataFrame, group: str) -> np.ndarray:
        """Calls predicted for one group, back on the original (non-log) scale."""
        return np.expm1(self.models[group].predict(df[self.feature_cols(group)]))

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Total predicted calls — every group summed. This is the number the tweet copy
        and the district map are built from, so it lives here rather than being
        re-derived by each consumer.
        """
        total = np.zeros(len(df), dtype=float)
        for group in self.models:
            total += self.predict_group(df, group)
        return total

    def __getitem__(self, group: str) -> XGBRegressor:
        return self.models[group]

    def __iter__(self):
        return iter(self.models)

    def __len__(self) -> int:
        return len(self.models)

    def items(self):
        return self.models.items()


class GroupedCallModelWrapper(mlflow.pyfunc.PythonModel):
    """mlflow.pyfunc face for GroupedCallModel — one registered model, one version.

    Registering each head separately made the version number meaningless: nothing tied
    version N of the noise model to version N of any other, though inference only ever
    uses all of them together. The deployment unit is the ensemble, so that is what gets
    a version.

    Unpickling in load_context needs modeling.pipelines.modeling.model importable in
    whatever environment loads the model, which is why log_grouped_run passes this
    module's package as code_paths.
    """

    def load_context(self, context) -> None:
        with open(context.artifacts["model"], "rb") as f:
            self.model = pickle.load(f)

    def predict(self, context, model_input: pd.DataFrame, params=None) -> np.ndarray:
        return self.model.predict(model_input)
