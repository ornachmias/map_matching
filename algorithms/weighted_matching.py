import math

from shapely import Point

from algorithms.matching_algorithm import MatchingAlgorithm
from dataset import project_edge_geometry


class WeightedMatching(MatchingAlgorithm):
    """
    This algorithm uses a weighted combination of distance and bearing difference to find the best matching edge.
    """
    search_radius = 150.0
    distance_weight = 0.5
    bearing_weight = 0.5

    @staticmethod
    def get_name() -> str:
        return 'Weighted Matching Algorithm'

    def get_candidates(self, projected_gps_point, accuracy=None) -> list:
        try:
            max_distance = self.search_radius if accuracy is None else accuracy * 3
            candidate_indices = list(self.index.query(projected_gps_point.buffer(max_distance)))

            if len(candidate_indices) == 0:
                raise Exception('No edge found in fallback matching.')

        except Exception as e:
            print(f'Error in spatial search: {e}. Falling back to nearest edge search.')
            candidate_indices = [self._get_nearest_candidate(projected_gps_point)]

        return candidate_indices

    def find_best_match(self, candidate_indices: list, point_bearing: float, point_accuracy: float, projected_gps_point: Point) -> int:
        max_distance = self.search_radius if point_accuracy is None else point_accuracy * 3
        best_score = -float('inf')
        best_match = None
        for edge_id in candidate_indices:
            try:
                edge = self.graph.es[edge_id]
                projected_segment = project_edge_geometry(edge['geometry'])
                distance = projected_gps_point.distance(projected_segment)

                if distance > max_distance:
                    continue

                match_score = self.get_match_score(distance, point_bearing, edge)
                if match_score > best_score:
                    best_score = match_score
                    best_match = edge_id
            except Exception:
                continue

        return best_match

    def get_match_score(self, distance, point_bearing, edge):
        distance_score = 1.0 / (1.0 + distance)
        bearing_diff = self._calculate_heading_difference(point_bearing, edge['bearing'])
        bearing_score = (math.cos(math.radians(bearing_diff)) + 1) / 2.0
        match_score = (self.distance_weight * distance_score) + (self.bearing_weight * bearing_score)
        return match_score


