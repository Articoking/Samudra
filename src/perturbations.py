from abc import ABC, abstractmethod

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
        # TODO
        datasets = []
        for member_idx in range(self.n_members):
            # TODO
            # Create perturbation of same size as ocean state tensor
            # Add it to first timestep of `dataset` and save result as InferenceDataset
            # Append it to `datasets`
            pass

        return InferenceDatasets(
            datasets,
            [dataset.__len__ for _ in range(self.n_members)], 
            )