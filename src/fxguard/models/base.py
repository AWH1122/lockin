from abc import ABC
from functools import partial

import lightning as L
import numpy as np
import torch
from lightning import Trainer
from torch.utils.data import DataLoader
from torchmetrics import MeanAbsolutePercentageError, MeanSquaredError
import torch.nn as nn
from torchmetrics.wrappers import MultioutputWrapper

from fxguard.data.timeseries import TimeSeries, TimeSeriesDataset

Batch = tuple[torch.Tensor, torch.Tensor, torch.Tensor]


def collate_dict(batch: list[dict]):
    out: dict[str, torch.Tensor | list | dict] = {}
    keys = batch[0].keys()
    for k in keys:
        vals = [b[k] for b in batch]
        v0 = vals[0]
        if torch.is_tensor(v0):
            out[k] = torch.stack(vals)
        elif isinstance(v0, np.ndarray):
            out[k] = torch.as_tensor(np.stack(vals))
        elif isinstance(v0, (float, int)):
            out[k] = torch.as_tensor(vals)
        else:
            out[k] = vals[0]  # weird I need to do this
    return out


class BaseLightning(L.LightningModule, ABC):
    def __init__(
        self,
        input_length: int,
        h: int | list[int] = 1,
        n_epochs: int = 100,
        batch_size: int = 32,
        optimizer_cls: torch.optim.Optimizer = torch.optim.Adam,
        optimizer_kwargs: dict | None = None,
        lr_scheduler_cls: torch.optim.lr_scheduler.LRScheduler | None = None,
        lr_scheduler_kwargs: dict | None = None,
        trainer_kwargs: dict | None = None,
        loss_fn=nn.MSELoss,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        h = list(range(1, h + 1)) if isinstance(h, int) else h
        trainer_kwargs = trainer_kwargs or {}
        self.trainer = partial(Trainer, max_epochs=n_epochs, **trainer_kwargs)
        optimizer_kwargs = optimizer_kwargs or {}
        lr_scheduler_kwargs = lr_scheduler_kwargs or {}
        self.criterion = loss_fn()
        self.save_hyperparameters()

    def configure_optimizers(self):
        optimizer = self.hparams.optimizer_cls(self.parameters(), **self.hparams.optimizer_kwargs)
        if self.hparams.lr_scheduler_cls is not None:
            return [optimizer], [self.hparams.lr_scheduler_cls(optimizer, **self.hparams.lr_scheduler_kwargs)]
        return optimizer

    def training_step(self, batch: Batch):
        x = batch['future_target']  # should be (batch_size,n_horizons,1)
        xhat = self(batch)
        loss = self.criterion(xhat, x)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch: Batch):
        x = batch['future_target']  # should be (batch_size,n_horizons,1)
        xhat = self(batch)
        loss = self.criterion(xhat, x)
        self.log('val_loss', loss)
        return loss

    def train_dataloader(self):
        return DataLoader(self.train_dataset, **self._dataloader_kwargs, collate_fn=collate_dict)

    def val_dataloader(self):
        if self.val_dataset is not None:
            return DataLoader(self.val_dataset, **self._dataloader_kwargs, collate_fn=collate_dict)
        return []

    def fit(
        self,
        series,
        past_covariates=None,
        future_covariates=None,
        val_series=None,
        val_past_covariates=None,
        val_future_covariates=None,
        epochs=None,
        dataloader_kwargs=None,
    ):
        self.trainer: Trainer
        self._dataloader_kwargs = dataloader_kwargs or {}
        if 'shuffle' not in self._dataloader_kwargs:
            self._dataloader_kwargs['shuffle'] = True
        if 'batch_size' not in self._dataloader_kwargs:
            self._dataloader_kwargs['batch_size'] = self.hparams.batch_size
        if epochs is not None:
            self.trainer = self.trainer(max_epochs=epochs)
        else:
            self.trainer = self.trainer()
        self.train_dataset = TimeSeriesDataset(
            series,
            input_length=self.hparams.input_length,
            h=self.hparams.h,
            past_covariates=past_covariates,
            future_covariates=future_covariates,
        )
        if val_series is not None:
            self.val_dataset = TimeSeriesDataset(
                val_series,
                input_length=self.hparams.input_length,
                h=self.hparams.h,
                past_covariates=val_past_covariates,
                future_covariates=val_future_covariates,
            )
        else:
            self.val_dataset = None
        self.trainer.fit(self)
        # return self

    def predict(
        self,
        series: TimeSeries | list[TimeSeries],
        past_covariates: TimeSeries | list[TimeSeries] | None = None,
        future_covariates: TimeSeries | list[TimeSeries] | None = None,
        batch_size: int | None = None,
        dataloader_kwargs: dict | None = None,
    ):
        dataloader_kwargs = dataloader_kwargs or {}
        if batch_size is not None:
            dataloader_kwargs['batch_size'] = batch_size
        last_chunks = [ts[: -self.hparams.input_length] for ts in series]
        predict_dataset = TimeSeriesDataset(
            last_chunks,
            input_length=self.hparams.input_length,
            h=None,
            past_covariates=past_covariates,
            future_covariates=future_covariates,
        )
        return self.trainer.predict(
            self,
            dataloaders=DataLoader(
                predict_dataset, collate_fn=collate_dict, **dataloader_kwargs
            ),  # need to remove dataloader
        )  # the shape here will have length equal to num input series

    def historical_forecast(
        self,
        series: TimeSeries | list[TimeSeries],
        past_covariates: TimeSeries | list[TimeSeries] | None = None,
        future_covariates: TimeSeries | list[TimeSeries] | None = None,
        stride: int = 1,
    ):
        series_list = series if isinstance(series, list) else [series]
        past_cov_list = (
            past_covariates if isinstance(past_covariates, list) or past_covariates is None else [past_covariates]
        )
        future_cov_list = (
            future_covariates
            if isinstance(future_covariates, list) or future_covariates is None
            else [future_covariates]
        )
        all_predictions = []
        for i, ts in enumerate(series_list):
            pcov = past_cov_list[i] if past_cov_list and len(past_cov_list) > 1 else past_covariates
            fcov = future_cov_list[i] if future_cov_list and len(future_cov_list) > 1 else future_covariates
            dataset = TimeSeriesDataset(
                ts,
                input_length=self.hparams.input_length,
                h=self.hparams.h,
                past_covariates=pcov,
                future_covariates=fcov,
                stride=stride,
            )
            dataloader = DataLoader(dataset, batch_size=self.hparams.batch_size, shuffle=False, collate_fn=collate_dict)
            predictions = []
            for batch in dataloader:
                predictions.append(self(batch).cpu().detach().numpy())
            all_predictions.append(
                TimeSeries(
                    times=ts.time_index[self.hparams.input_length : len(ts.time_index) - max(self.hparams.h) + 1],
                    values=np.concat(predictions, axis=0),
                    static_covariates=ts.static_covariates,
                    components=[f'H{horizon}' for horizon in self.hparams.h],
                )
            )
        return all_predictions if len(all_predictions) > 1 else all_predictions[0]
