#!/usr/bin/env python3
"""Cache only deterministic dataset preparation in Paper1 TabDDPM sampling.

``PreparedPaper1Sampler`` is a deliberately narrow adaptation of
``diffusion/tabddpm_official/scripts/sample.py``.  Its first call follows the
native function in the same order.  Later calls reuse only:

* the transformed training/validation/test dataset, and
* the raw training features used to identify integer-valued columns.

The model, checkpoint, diffusion object, requested conditions, random seed,
generated samples, and all downstream acceptance decisions are rebuilt or
evaluated on every call.  This removes repeated reads and QuantileTransformer
fits without caching any stochastic result.
"""

from dataclasses import dataclass
import os
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class _PreparationKey:
    """Inputs that determine the prepared Paper1 dataset."""

    real_data_path: str
    transformations: tuple
    num_classes: int
    is_y_cond: bool
    change_val: bool


class PreparedPaper1Sampler:
    """Paper1 ``sample`` callable with a per-model in-memory data cache."""

    def __init__(self, paper1_module):
        self.module = paper1_module
        self._key = None
        self._dataset = None
        self._x_num_real = None
        self._discrete_columns_cache = None
        self.dataset_preparations = 0
        self.discrete_column_scans = 0

    @staticmethod
    def _preparation_key(real_data_path, T_dict, model_params, change_val):
        transformations = tuple(sorted(T_dict.items()))
        return _PreparationKey(
            real_data_path=str(Path(real_data_path).expanduser().resolve()),
            transformations=transformations,
            num_classes=int(model_params["num_classes"]),
            is_y_cond=bool(model_params["is_y_cond"]),
            change_val=bool(change_val),
        )

    def _dataset_for(self, real_data_path, T, T_dict, model_params, change_val):
        key = self._preparation_key(real_data_path, T_dict, model_params, change_val)
        if self._key is None:
            self._key = key
            self._dataset = self.module.make_dataset(
                real_data_path,
                T,
                num_classes=model_params["num_classes"],
                is_y_cond=model_params["is_y_cond"],
                change_val=change_val,
            )
            self.dataset_preparations += 1
        elif key != self._key:
            raise ValueError(
                "A PreparedPaper1Sampler cannot be reused across different datasets "
                "or transformation configurations"
            )
        return self._dataset

    def _raw_numerical_features(self, real_data_path):
        if self._x_num_real is None:
            self._x_num_real = np.load(
                os.path.join(real_data_path, "X_num_train.npy"), allow_pickle=True
            )
        return self._x_num_real

    def _discrete_columns(self, real_data_path):
        if self._discrete_columns_cache is None:
            X_num_real = self._raw_numerical_features(real_data_path)
            columns = []
            for col in range(X_num_real.shape[1]):
                uniq_vals = np.unique(X_num_real[:, col])
                if len(uniq_vals) <= 32 and (
                    (uniq_vals - np.round(uniq_vals)) == 0
                ).all():
                    columns.append(col)
            self._discrete_columns_cache = tuple(columns)
            self.discrete_column_scans += 1
            # round_columns needs the raw reference only when discrete columns
            # exist.  The COUNT tracker features are continuous, so release the
            # otherwise redundant array after its one deterministic scan.
            if not columns:
                self._x_num_real = None
        return self._discrete_columns_cache

    def __call__(
        self,
        parent_dir,
        real_data_path="data/higgs-small",
        batch_size=2000,
        num_samples=0,
        model_type="mlp",
        model_params=None,
        model_path=None,
        num_timesteps=1000,
        gaussian_loss_type="mse",
        scheduler="cosine",
        T_dict=None,
        num_numerical_features=0,
        disbalance=None,
        device=None,
        seed=0,
        change_val=False,
        y_to_sample=None,
    ):
        """Run the native sampling algorithm while reusing prepared data."""
        module = self.module
        torch = module.torch

        # Keep seeding at the native entry point.  Dataset preparation uses the
        # explicit T_dict seed; model construction and diffusion sampling see
        # the same per-call global random state as the uncached implementation.
        module.zero.improve_reproducibility(seed)

        T = module.lib.Transformations(**T_dict)
        D = self._dataset_for(real_data_path, T, T_dict, model_params, change_val)

        K = np.array(D.get_category_sizes("train"))
        if len(K) == 0 or T_dict["cat_encoding"] == "one-hot":
            K = np.array([0])

        num_numerical_features_ = D.X_num["train"].shape[1] if D.X_num is not None else 0
        d_in = np.sum(K) + num_numerical_features_
        model_params["d_in"] = int(d_in)
        model = module.get_model(
            model_type,
            model_params,
            num_numerical_features_,
            category_sizes=D.get_category_sizes("train"),
        )

        model.load_state_dict(torch.load(model_path, map_location="cpu"))

        diffusion = module.GaussianMultinomialDiffusion(
            K,
            num_numerical_features=num_numerical_features_,
            denoise_fn=model,
            num_timesteps=num_timesteps,
            gaussian_loss_type=gaussian_loss_type,
            scheduler=scheduler,
            device=device,
        )
        diffusion.to(device)
        diffusion.eval()

        _, empirical_class_dist = torch.unique(
            torch.from_numpy(np.concatenate([D.y["train"], D.y["val"]])),
            return_counts=True,
        )
        if y_to_sample is not None:
            y_to_sample = np.asarray(y_to_sample)
            x_gen, y_gen = diffusion.sample_all(
                len(y_to_sample),
                batch_size,
                empirical_class_dist.float(),
                ddim=False,
                y_explicit=y_to_sample,
            )
        elif disbalance == "fix":
            empirical_class_dist[0], empirical_class_dist[1] = (
                empirical_class_dist[1], empirical_class_dist[0]
            )
            x_gen, y_gen = diffusion.sample_all(
                num_samples, batch_size, empirical_class_dist.float(), ddim=False
            )
        elif disbalance == "fill":
            ix_major = empirical_class_dist.argmax().item()
            val_major = empirical_class_dist[ix_major].item()
            x_gen, y_gen = [], []
            for index in range(empirical_class_dist.shape[0]):
                if index == ix_major:
                    continue
                distribution = torch.zeros_like(empirical_class_dist)
                distribution[index] = 1
                missing = val_major - empirical_class_dist[index].item()
                x_temp, y_temp = diffusion.sample_all(
                    missing, batch_size, distribution.float(), ddim=False
                )
                x_gen.append(x_temp)
                y_gen.append(y_temp)
            x_gen = torch.cat(x_gen, dim=0)
            y_gen = torch.cat(y_gen, dim=0)
        else:
            x_gen, y_gen = diffusion.sample_all(
                num_samples, batch_size, empirical_class_dist.float(), ddim=False
            )

        X_gen, y_gen = x_gen.numpy(), y_gen.numpy()
        num_numerical_features += int(
            D.is_regression and not model_params["is_y_cond"]
        )

        X_num_ = X_gen
        if num_numerical_features < X_gen.shape[1]:
            np.save(
                os.path.join(parent_dir, "X_cat_unnorm"),
                X_gen[:, num_numerical_features:],
            )
            if T_dict["cat_encoding"] == "one-hot":
                X_gen[:, num_numerical_features:] = module.to_good_ohe(
                    D.cat_transform.steps[0][1],
                    X_num_[:, num_numerical_features:],
                )
            X_cat = D.cat_transform.inverse_transform(
                X_gen[:, num_numerical_features:]
            )

        if num_numerical_features_ != 0:
            np.save(
                os.path.join(parent_dir, "X_num_unnorm"),
                X_gen[:, :num_numerical_features],
            )
            X_num_ = D.num_transform.inverse_transform(
                X_gen[:, :num_numerical_features]
            )
            X_num = X_num_[:, :num_numerical_features]

            # The native sampler reloads this unchanged array on every call.
            # Keeping it in memory changes no rounding inputs or decisions.
            disc_cols = self._discrete_columns(real_data_path)
            print("Discrete cols:", list(disc_cols))
            if model_params["num_classes"] == 0:
                y_gen = X_num[:, 0]
                X_num = X_num[:, 1:]
            if disc_cols:
                X_num = module.round_columns(self._x_num_real, X_num, disc_cols)

        if num_numerical_features != 0:
            print("Num shape: ", X_num.shape)
            np.save(os.path.join(parent_dir, "X_num_train"), X_num)
        if num_numerical_features < X_gen.shape[1]:
            np.save(os.path.join(parent_dir, "X_cat_train"), X_cat)
        np.save(os.path.join(parent_dir, "y_train"), y_gen)
