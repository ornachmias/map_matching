from shapely import Point

from algorithms.weighted_matching import WeightedMatching


class TopologicalMatching(WeightedMatching):
    """
    An incremental topological map matcher that maintains state between GPS points.
    Uses topology to constrain candidate edges based on the previous match.
    """

    def __init__(self, graph):
        super().__init__(graph)
        self.last_match = None  # Track the previous match for topological constraints

    @staticmethod
    def get_name() -> str:
        return 'Incremental Topological Algorithm'

    def reset_route(self):
        self.last_match = None

    def topology_bias(self, edge):
        topology_score = 0.  # Base score
        if self.last_match is not None:
            try:
                prev_edge = self.graph.es[self.last_match]
                if edge.index == self.last_match:
                    # Same edge, strong bonus
                    topology_score = 0.2
                elif (prev_edge.source in [edge.source, edge.target] or
                      prev_edge.target in [edge.source, edge.target]):
                    # Strong bonus for connected edges
                    topology_score = 0.1
            except Exception as e:
                print(f"Error in topology factor calculation: {e}")

        return topology_score

    def get_match_score(self, distance, point_bearing, edge):
        base_score = super().get_match_score(distance, point_bearing, edge)
        topology_score = self.topology_bias(edge) + base_score
        return topology_score

    def find_best_match(self, candidate_indices: list, point_bearing: float, point_accuracy: float,
                        projected_gps_point: Point) -> int:
        best_edge_id = super().find_best_match(candidate_indices, point_bearing, point_accuracy, projected_gps_point)
        if best_edge_id is not None:
            self.last_match = best_edge_id
        return best_edge_id

