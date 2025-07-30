from darts.dataprocessing.transformers import Scaler as ds
from sklearn.preprocessing import MinMaxScaler
from fxguard.data.timeseries import TimeSeries
from sklearn.base import BaseEstimator, TransformerMixin

class Scaler(BaseEstimator, TransformerMixin):
    def __init__(self, scaler_cls=MinMaxScaler, **scaler_kwargs):
        self.scaler_cls = scaler_cls
        self.scaler_kwargs = scaler_kwargs

    def fit(self, X: TimeSeries | list[TimeSeries], y=None):
        if not isinstance(X, list):
            X = [X]
        self._scalers = []
        for s in X:
            scaler = self.scaler_cls(**self.scaler_kwargs)
            scaler.fit(s.values)
            self._scalers.append(scaler)
        return self

    def transform(
        self, X: TimeSeries | list[TimeSeries], series_idx: int | list[int] | None = None
    ) -> TimeSeries | list[TimeSeries]:
        series_idx = series_idx or range(len(X))
        if isinstance(series_idx, int):
            series_idx = [series_idx]
        transformed = []
        for i in series_idx:
            s = X[i].copy()
            s.values = self._scalers[i].transform(s.values)
            transformed.append(s)
        return transformed

    def inverse_transform(
        self, X: TimeSeries | list[TimeSeries], series_idx: int | list[int] | None = None
    ) -> TimeSeries | list[TimeSeries]:
        series_idx = series_idx or range(len(X))
        if isinstance(series_idx, int):
            series_idx = [series_idx]
        transformed = []
        for i in series_idx:
            s = X[i].copy()
            s.values = self._scalers[i].inverse_transform(s.values)
            transformed.append(s)
        return transformed
