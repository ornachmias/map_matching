from shapely import Point

from algorithms.matching_algorithm import MatchingAlgorithm


class Baseline(MatchingAlgorithm):
    def __init__(self, graph):
        super().__init__(graph)

    @staticmethod
    def get_name() -> str:
        return 'Baseline Matching Algorithm'

    def get_candidates(self, projected_gps_point, accuracy=None):
        return [self._get_nearest_candidate(projected_gps_point)]

    def find_best_match(self, candidate_indices: list, point_bearing: float, point_accuracy: float, projected_gps_point: Point) -> int:
        return candidate_indices[0]


