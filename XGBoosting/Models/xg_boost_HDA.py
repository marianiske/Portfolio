from xgboost import XGBClassifier
from xgboost.callback import EarlyStopping


class XGBMatchModel:
    def __init__(self):
        self.model = self._build_model()

    def _build_model(self):
        return XGBClassifier(
            objective="multi:softprob",
            num_class=3,
            n_estimators=5000,
            learning_rate=0.03,
            max_depth=4,
            min_child_weight=3,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            eval_metric="mlogloss",
            tree_method="hist",
            random_state=42,
            n_jobs=-1,
            callbacks=[EarlyStopping(rounds=100, save_best=True)],
        )

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        fit_params = {}
        if X_val is not None and y_val is not None:
            fit_params["eval_set"] = [(X_val, y_val)]

        self.model.fit(X_train, y_train, verbose=False, **fit_params)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)
