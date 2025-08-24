from abc import ABC, abstractmethod

from pyproj import CRS, Transformer
from shapely import STRtree
from shapely.geometry import Point, LineString
from shapely.ops import nearest_points

import config

class MatchingAlgorithm(ABC):
    # Set up coordinate transformers like in dataset.py
    input_crs = CRS.from_user_input(config.default_crs)
    output_crs = CRS.from_user_input(config.local_crs)
    transformer = Transformer.from_crs(input_crs, output_crs, always_xy=True)
    reversed_transformer = Transformer.from_crs(output_crs, input_crs, always_xy=True)

    def __init__(self, graph):
        self.graph = graph
        self.index = self._get_projected_edges_index(graph)

    def reset_route(self):
        """
        Resets any stateful information for a new route.
        """
        pass

    @abstractmethod
    def get_candidates(self, projected_gps_point, accuracy=None) -> list:
        """
        Abstract method to get candidate edges for a given projected GPS point.
        """
        pass

    @abstractmethod
    def find_best_match(self, candidate_indices: list, point_bearing: float, point_accuracy: float, projected_gps_point: Point) -> int:
        """
        Abstract method to find the best match among candidate edges.
        """
        pass

    @staticmethod
    def _extract_sample(sample: dict) -> tuple:
        """
        Extracts longitude, latitude, bearing, and accuracy from a sample dictionary.
        """
        lon = sample['noisy']['longitude']
        lat = sample['noisy']['latitude']
        bearing = sample['noisy']['bearing']
        accuracy = sample['noisy'].get('accuracy')
        return lon, lat, bearing, accuracy

    def _get_projected_edges_index(self, graph):
        """
        Builds a projected STRtree index for the graph edges to optimize distance calculations.
        """
        projected_geometries = [self._project_edge(edge['geometry']) for edge in graph.es]
        return STRtree(projected_geometries)

    def _project_edge(self, edge: LineString) -> LineString:
        """
        Projects the edge geometry to the output CRS for accurate distance calculations.
        """
        projected_coords = [self.transformer.transform(x, y) for x, y in zip(edge.xy[0], edge.xy[1])]
        return LineString(projected_coords)

    def _project_point(self, point: Point) -> Point:
        """
        Projects a point to the output CRS for accurate distance calculations.
        """
        return self._project_coordinates(point.x, point.y)

    def _project_coordinates(self, longitude: float, latitude: float) -> Point:
        """
        Projects GPS coordinates to the output CRS for accurate distance calculations.
        """
        projected_x, projected_y = self.transformer.transform(longitude, latitude)
        return Point(projected_x, projected_y)

    def _get_nearest_candidate(self, projected_gps_point: Point) -> tuple:
        """
        Get candidate edges using spatial search with the projected STRtree index.
        """
        candidate_index = self.index.query_nearest(projected_gps_point).tolist()[0]
        return candidate_index

    @staticmethod
    @abstractmethod
    def get_name() -> str:
        """
        Returns the name of the algorithm.
        """
        pass

    @staticmethod
    def _project_point_to_segment(point: Point, segment: LineString) -> tuple:
        """
        Projects a point onto a line segment and returns the projected point,
        the ratio along the segment, and the distance.

        Args:
            point (Point): The point to project
            segment (LineString): The line segment

        Returns:
            tuple: (projected_point, ratio, distance)
        """
        # Find the nearest point on the segment
        projected_point = nearest_points(segment, point)[0]

        # Calculate the distance from the original point to the projected point
        distance = point.distance(projected_point)

        # Calculate the ratio along the segment (0 = start, 1 = end)
        if segment.length == 0:
            ratio = 0.0
        else:
            ratio = segment.project(projected_point) / segment.length

        return projected_point, ratio, distance

    @staticmethod
    def _calculate_heading_difference(heading1: float, heading2: float) -> float:
        """
        Calculates the absolute difference between two headings, considering
        the circular nature of angles (e.g., 350° and 10° are 20° apart).

        Args:
            heading1 (float): First heading in degrees (0-360)
            heading2 (float): Second heading in degrees (0-360)

        Returns:
            float: Absolute difference in degrees (0-180)
        """
        diff = abs(heading1 - heading2)
        if diff > 180:
            diff = 360 - diff
        return diff

    def match_sample(self, sample: dict):
        """
        Runs the matching algorithm for a single sample and returns the prediction dict.
        """
        lon, lat, bearing, accuracy = self._extract_sample(sample)
        projected_gps_point = self._project_coordinates(lon, lat)
        candidate_indices = self.get_candidates(projected_gps_point, accuracy)
        matched_edge_index = self.find_best_match(candidate_indices, bearing, accuracy, projected_gps_point)
        if matched_edge_index is None:
            snapped_lon, snapped_lat = lon, lat
        else:
            projected_edge = self._project_edge(self.graph.es[matched_edge_index]['geometry'])
            projected_snapped_point, ratio, _ = self._project_point_to_segment(projected_gps_point, projected_edge)
            snapped_lon, snapped_lat = self.reversed_transformer.transform(projected_snapped_point.x, projected_snapped_point.y)
        return {
            'edge_id': matched_edge_index,
            'longitude': snapped_lon,
            'latitude': snapped_lat,
        }
