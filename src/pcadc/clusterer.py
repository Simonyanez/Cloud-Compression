import numpy as np
from src.pcadc.objects import *
from src.pcadc.color import *
from dataclasses import dataclass


@dataclass
class Codebook:
    pass

class Clusterer:
    def __init__(self, n_clusters: int):
        self.n_clusters = n_clusters

    def __call__(self, fit_collection: FitCollection):
        slope_matrix = self._init_slope_matrix(fit_collection)
        pass

    def _init_slope_matrix(self, fit_collection: FitCollection)
    
