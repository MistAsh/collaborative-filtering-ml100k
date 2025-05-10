from abc import ABC

from utils.similarity import SimilarityComputer
from cf.base import BaseCF



class SimilarityBasedCF(BaseCF, ABC):

    def __init__(
            self,
            similarity_threshold: float,
            similarity_computer: SimilarityComputer
    ):
        super().__init__()
        self.similarity_threshold = similarity_threshold
        self.similarity_computer = similarity_computer
        self.average_neighbors_per_user = None
