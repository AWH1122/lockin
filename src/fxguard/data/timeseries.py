from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
from numpy.typing import ArrayLike
from torch.utils.data import Dataset


class TimeSeries:
    def __init__(
        self,
        times: pd.DatetimeIndex,
        values: ArrayLike,
        components: pd.Index | Sequence | None = None,
        static_covariates: dict[str, Any] | None = None,
        freq: str = 'D',
    ):
        if not isinstance(values, np.ndarray):
            values = np.asarray(values)
        if not isinstance(times, pd.DatetimeIndex):
            times = pd.to_datetime(times, utc=True)
        self.values = values
        self.time_index = times
        if len(values) != len(times):
            raise ValueError(f'times and values must have the same length, got {len(times)} and {len(values)}')
        if components is None:
            components = pd.Index([str(idx) for idx in range(values.shape[-1])])
        self._components = components
        self.static_covariates = static_covariates or {}
        self.freq = freq

    def __getitem__(self, idx):
        # If idx is a string, return a new TimeSeries for that component
        if isinstance(idx, str):
            col_idx = (
                self._components.get_loc(idx) if hasattr(self._components, 'get_loc') else self._components.index(idx)
            )
            vals = self.values[:, col_idx]
            return TimeSeries(self.time_index, vals, [idx], static_covariates=self.static_covariates, freq=self.freq)
        # Otherwise, slice by time
        return TimeSeries(
            self.time_index[idx],
            self.values[idx],
            self._components,
            static_covariates=self.static_covariates,
            freq=self.freq,
        )

    def __len__(self):
        return len(self.values)

    def to_series(self):
        # Only works for single-component series
        if len(self._components) != 1:
            raise ValueError('to_series only works for single-component series')
        return pd.Series(self.values.flatten(), index=self.time_index, name=self._components[0])

    def to_dataframe(self):
        return pd.DataFrame(self.values, index=self.time_index, columns=self._components)

    @property
    def columns(self):
        return self._components

    def plot(self, **kwargs):
        return self.to_dataframe().plot(**kwargs)

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        time_col: str | None = None,
        value_cols: str | list[str] | None = None,
        static_cols: str | list[str] | None = None,
        freq: str = 'D',
    ):
        df = df.copy().asfreq(freq).ffill()
        time_col = time_col or ''
        index = df[time_col] if time_col != '' else df.index
        index = pd.to_datetime(index, utc=True)
        static_cols = static_cols or []
        static_covariates = df[static_cols].to_dict()
        if isinstance(value_cols, str):
            value_cols = [value_cols]
        else:
            value_cols = value_cols or df.columns.drop(static_cols + [time_col])
        values = df[value_cols].values
        return cls(index, values, value_cols, static_covariates, freq)

    @classmethod
    def from_group_dataframe(
        cls,
        df: pd.DataFrame,
        group_col: str,
        time_col: str | None = None,
        value_cols: str | list[str] | None = None,
        static_cols: list[str] | None = None,
        freq: str = 'D',
    ) -> list['TimeSeries']:
        df = df.copy()
        time_col = time_col or ''
        index = df[time_col] if time_col != '' else df.index
        index = pd.to_datetime(index, utc=True)
        static_cols = static_cols or []
        static_cols.append(group_col)
        if isinstance(value_cols, str):
            value_cols = [value_cols]
        else:
            value_cols = value_cols or df.columns.drop(
                [group_col, time_col, *static_cols]
            )  # in my case should be singleton when making series for target. not for covariates.
        out: list[TimeSeries] = []
        for gid, g in df.groupby(by=group_col, sort=False):
            g = g.sort_values(time_col).set_index(time_col)
            static = g[static_cols].iloc[0].to_dict()
            out.append(cls(g.index, g[value_cols].values, value_cols, static, freq))
        return out


class TimeSeriesDataset(Dataset):
    def __init__(
        self,
        ts: TimeSeries | list[TimeSeries],
        input_length: int,
        h: int | list[int] | None,
        past_covariates: TimeSeries | list[TimeSeries] = None,
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
        self.shifts = {}
        for s_idx, s in enumerate(self.series_list):
            n = len(s)
            min_date = s.time_index.min()
            try:
                past_shift = past_covariates.time_index.get_loc(min_date)
            except (KeyError, AttributeError) as e:
                if type(e) is KeyError:
                    raise e
                past_shift = 0
            try:
                future_shift = future_covariates.time_index.get_loc(min_date)
            except (KeyError, AttributeError) as e:
                if type(e) is KeyError:
                    raise e
                future_shift = 0
            self.shifts[s_idx] = [past_shift, future_shift]
            for end in range(self.L, n - self.H + 1, stride):
                self.index.append((s_idx, end))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx) -> dict[str, dict | torch.Tensor]:
        s_idx, end = self.index[idx]
        ts = self.series_list[s_idx]
        past_slice = slice(end - self.L, end)
        future_slice = slice(end, end + self.H)

        past_target = torch.tensor(ts.values[past_slice], dtype=torch.float32)
        future_target = torch.tensor(ts.values[future_slice].flatten(), dtype=torch.float32)

        def get_covariate_tensor(covariates, slc, shift):
            if covariates is None:
                return None
            aligned = slice(slc.start + shift, slc.stop + shift)
            vals = covariates[aligned].values
            return torch.tensor(vals, dtype=torch.float32).unsqueeze(-1)

        past_shift = self.shifts[s_idx][0]
        future_shift = self.shifts[s_idx][1]
        past_covariates = get_covariate_tensor(self.past_covariates, past_slice, past_shift)
        historic_future_covariates = get_covariate_tensor(self.future_covariates, past_slice, past_shift)
        future_covariates = get_covariate_tensor(self.future_covariates, future_slice, future_shift)

        out = {
            'past_target': past_target,
            'past_covariates': past_covariates,
            'historic_future_covariates': historic_future_covariates,
            'future_covariates': future_covariates,
            'static_covariates': ts.static_covariates,
            'future_target': future_target,
        }
        return out
