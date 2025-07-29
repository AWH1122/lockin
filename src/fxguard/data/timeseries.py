import pandas as pd
import numpy as np
import torch
from typing import Any
from torch.utils.data import Dataset


class TimeSeries:
    def __init__(
        self,
        times: pd.Index,
        values: np.ndarray,
        static_covariates: dict[str, Any] | None = None,
        freq: str = 'D',
    ):
        self.data = pd.DataFrame(values, times).asfreq(freq).ffill()
        self.dates = self.data.index
        self.values = self.data.values
        self.static_covariates = static_covariates or {}

    def __getitem__(self, idx):
        return TimeSeries(self.dates[idx], self.values[idx], static_covariates=self.static_covariates)

    def __len__(self):
        return len(self.values)

    def to_series(self):
        return pd.Series(self.values, self.dates)

    def to_df(self):
        return pd.DataFrame(self.values, self.dates)

    def plot(self, **kwargs):
        return self.data.plot(**kwargs)

    @classmethod
    def from_group_dataframe(
        cls,
        df: pd.DataFrame,
        group_col: str,
        time_col: str | None = None,
        target_cols: str | list[str] | None = None,
        static_cols: list[str] | None = None,
        freq: str = 'D',
    ) -> list['TimeSeries']:
        df = df.copy()
        time_col = time_col or df.index
        static_cols = static_cols or []
        static_cols.append(group_col)
        target_cols = (
            target_cols or df.columns.drop([group_col, time_col, *static_cols]).values
        )  # in my case should be singleton when making series for target. not for covariates.
        if isinstance(target_cols, str):
            target_cols = [target_cols]
        df[time_col] = pd.to_datetime(df[time_col], utc=True)
        out: list[TimeSeries] = []
        for gid, g in df.groupby(by=group_col, sort=False):
            g = g.sort_values(time_col).set_index(time_col)
            static = g[static_cols].iloc[0].to_dict()
            out.append(cls(g.index, g[target_cols].values, static, freq))
        return out


class TimeSeriesDataset(Dataset):
    def __init__(
        self,
        ts: TimeSeries | list[TimeSeries],
        input_length: int,
        h: int | list[int] | None,
        past_covariates: TimeSeries | list[TimeSeries] = None,  # make sure past and future covariates line up properly
        future_covariates: TimeSeries | list[TimeSeries] = None,
        stride: int = 1,
    ):
        self.series_list = ts if isinstance(ts, list) else [ts]
        self.L = input_length
        h = h or 0
        self.horizons = list(range(1, h + 1)) if isinstance(h, int) else h
        self.H = max(self.horizons)

        self.past_covariates = past_covariates
        self.future_covariates = future_covariates

        self.index = []
        for s_idx, s in enumerate(self.series_list):
            n = len(s)
            for end in range(self.L, n - self.H + 1, stride):
                self.index.append((s_idx, end))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx) -> dict[str, dict | torch.Tensor]:
        s_idx, end = self.index[idx]
        ts = self.series_list[s_idx]
        past_slice = slice(end - self.L, end)
        future_slice = slice(end, end + self.H)

        past_target = torch.tensor(ts.data.iloc[past_slice].values, dtype=torch.float32)
        future_target = torch.tensor(ts.data.iloc[future_slice].values.flatten(), dtype=torch.float32)

        past_covariates = self.past_covariates.iloc[past_slice].to_dict() if self.past_covariates is not None else None
        historic_future_covariates = (
            self.future_covariates.iloc[past_slice].to_dict() if self.future_covariates is not None else None
        )
        future_covariates = (
            (self.future_covariates[future_slice].iloc[self.horizons].to_dict())
            if self.future_covariates is not None
            else None
        )
        out = {
            'past_target': past_target,
            'past_covariates': past_covariates,
            'historic_future_covariates': historic_future_covariates,
            'future_covariates': future_covariates,
            'static_covariates': ts.static_covariates,
            'future_target': future_target,
        }
        return out
