from algorithms.topological_matching import TopologicalMatching
from algorithms.weighted_matching import WeightedMatching
from algorithms.matching_algorithm import MatchingAlgorithm
import config

class HybridMatching(MatchingAlgorithm):
    """
    Hybrid matching algorithm that selects between WeightedMatching and TopologicalMatching
    based on the GPS accuracy value for each sample.
    """
    def __init__(self, graph):
        super().__init__(graph)
        self.weighted_matcher = WeightedMatching(graph)
        self.topological_matcher = TopologicalMatching(graph)
        self.last_accuracy_range = (config.accuracy_ranges[-2][0], config.accuracy_ranges[-1][1])

    @staticmethod
    def get_name():
        return "Hybrid Matching Algorithm"

    def match_sample(self, sample):
        accuracy = sample['actual'].get('accuracy', None)
        # We run both matchers to keep their state updated
        topology_result = self.topological_matcher.match_sample(sample)
        weighted_result = self.weighted_matcher.match_sample(sample)
        if accuracy is not None and self.last_accuracy_range[0] <= accuracy <= self.last_accuracy_range[1]:
            return topology_result
        else:
            return weighted_result

    def get_candidates(self, projected_gps_point, accuracy=None):
        # Not used in HybridMatching, but required by base class
        return []

    def find_best_match(self, candidate_indices, point_bearing, point_accuracy, projected_gps_point):
        # Not used in HybridMatching, but required by base class
        return None

    def reset_route(self):
        self.weighted_matcher.reset_route()
        self.topological_matcher.reset_route()


