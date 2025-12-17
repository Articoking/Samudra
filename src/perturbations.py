from abc import ABC, abstractmethod
import logging
import copy

import torch
import numpy as np
import xarray as xr

from datasets import InferenceDataset, InferenceDatasets
from config import EnsembleConfig

class Perturbator(ABC):
    def __init__(self, cfg: EnsembleConfig):
        self.n_members = cfg.n_members

    @property
    @abstractmethod
    def type(self):
        pass

    @abstractmethod
    def apply_perturbation(self, dataset: InferenceDataset) -> InferenceDatasets:
        pass

class GaussianNoisePerturbator(Perturbator):
    def __init__(self, cfg: EnsembleConfig):
        super().__init__(cfg)
        self.noise_level = cfg.perturbation.noise_level
    
    @property
    def type(self) -> str:
        return "GaussianNoise"
    
    def apply_perturbation(self, dataset: InferenceDataset) -> InferenceDatasets:
        """
        Generate perturbed versions of `dataset` by adding Gaussian noise to initial state.

        Parameters
        ----------
        dataset : InferenceDataset
            Unperturbed data from which to generate the perturbed versions

        Returns
        -------
        InferenceDataset
            List-like object containing each perturbed version of `dataset`
        """
        datasets = []
        initial_state = dataset._prognostic_data.loc[
            dict(time=dataset._prognostic_data.time[0])
            ]
        for _ in range(self.n_members):
            # Init perturbed InferenceDataset
            perturbed_ids = copy.copy(dataset)
            
            # Create perturbation of same size as ocean state tensor
            sigma = self.noise_level # TODO: Adapt sigma to each variable
            noise = xr.Dataset(
                {
                    var: xr.DataArray(
                        np.random.normal(0, sigma, size=initial_state[var].shape),
                        dims=initial_state[var].dims,
                        coords=initial_state[var].coords,
                    )
                    for var in initial_state.data_vars
                }
            )

            # Add it to first timestep
            perturbed_ids._prognostic_data.loc[
                dict(time=perturbed_ids._prognostic_data.time[0])
                ] += noise

            # Append it to `datasets`
            datasets.append(perturbed_ids)

        return InferenceDatasets(
            datasets,
            [dataset.__len__ for _ in range(self.n_members)], 
            )