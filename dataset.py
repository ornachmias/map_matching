import os
import random

import igraph
from igraph import Graph
from pyproj import CRS, Transformer
from shapely import line_interpolate_point, LineString

from config import Config

class Dataset:
    def __init__(self):
        self.dataset_dir = os.path.join(Config.output_dir, 'dataset')
        os.makedirs(self.dataset_dir, exist_ok=True)

        input_crs = CRS.from_user_input(Config.default_crs)
        output_crs = CRS.from_user_input(Config.local_crs)
        self.transformer = Transformer.from_crs(input_crs, output_crs, always_xy=True)
        self.reversed_transformer = Transformer.from_crs(output_crs, input_crs, always_xy=True)

    def get_dataset(self):
        pass

    def get_samples(self, graph: Graph, n_routes: int):
        samples = []
        for route_id in range(n_routes):
            source_vertex, target_vertex = self.get_random_vertices_pair(graph)
            vertices_route = graph.get_shortest_path(source_vertex, target_vertex, output='vpath', mode=igraph.OUT)

            time_delta = 0
            route_samples = []
            for route_step in range(len(vertices_route)):
                step_source = vertices_route[route_step]
                step_target = vertices_route[route_step + 1] if route_step + 1 < len(vertices_route) else None
                if step_target is None:
                    break

                edge_samples = self.edge_samples(graph, step_source, step_target, time_delta, route_id)
                if len(edge_samples) == 0:
                    continue

                time_delta += edge_samples[-1]['timestamp'] - time_delta
                route_samples.extend(edge_samples)

            samples.append(route_samples)

        return samples

    def edge_samples(self, graph, source_vertex, target_vertex, start_time, route_id):
        edge_index = graph.get_eid(source_vertex, target_vertex, directed=True)
        edge = graph.es[edge_index]
        edge_geometry = self.project_edge_geometry(edge['geometry'])

        total_distance = 0
        total_time = 0
        samples = []
        while total_distance < edge['length']:
            # Generate a random time interval, with minimum of 1 second
            time_interval = max([1, int(random.normalvariate(Config.mean_sampling_interval, Config.sigma_sampling_interval))])

            # Randomize the current speed based on the max speed
            current_speed = random.normalvariate(edge['maxspeed'], edge['maxspeed'] * Config.max_speed_factor)

            # Ensure the speed is reasonable
            current_speed = max([1, current_speed])

            # Convert km/h to m/s
            current_speed = current_speed / 3.6

            # Calculate the distance covered in that time interval
            distance_covered = current_speed * time_interval

            # Check if the distance is less than the edge length
            if total_distance + distance_covered > edge['length']:
                break

            accuracy = self.get_random_accuracy()

            # Get the coordinates of the current location
            coords = line_interpolate_point(edge_geometry, total_distance + distance_covered)

            noise_long, noise_lat = self.get_coordinates_noise(accuracy)
            noisy_cords = (coords.xy[0][0] + noise_long, coords.xy[1][0] + noise_lat)
            noisy_cords = self.reversed_transformer.transform(noisy_cords[0], noisy_cords[1])
            coords = self.reversed_transformer.transform(coords.xy[0][0], coords.xy[1][0])

            total_time += time_interval
            total_distance += distance_covered

            # Create a sample
            samples.append({
                'timestamp': start_time + total_time,
                'edge_index': edge_index,
                'route_id': route_id,
                'actual': {
                    'longitude': coords[0],
                    'latitude': coords[1],
                    'bearing': edge['bearing'],
                    'speed': current_speed,
                    'accuracy': accuracy
                },
                'noisy': {
                    'longitude': noisy_cords[0],
                    'latitude': noisy_cords[1],
                    'bearing': self.generate_noisy_bearing(edge['bearing'], current_speed),
                    'speed': current_speed,
                    'accuracy': accuracy
                }
            })

        return samples

    @staticmethod
    def get_random_accuracy():
        accuracy_range = random.choices(Config.accuracy_ranges, weights=Config.accuracy_weights, k=1)[0]
        accuracy = random.uniform(accuracy_range[0], accuracy_range[1])
        return accuracy

    @staticmethod
    def get_coordinates_noise(accuracy):
        noise_long = random.normalvariate(0, accuracy)
        noise_lat = random.normalvariate(0, accuracy)
        return noise_long, noise_lat

    @staticmethod
    def generate_noisy_bearing(true_bearing, speed):
        min_speed_threshold = 1.5  # m/s
        min_sigma_deg = 2.0  # Min noise std dev at high speeds
        base_sigma = 30.0  # Noise scaling factor (degrees * m/s)
        epsilon = 0.1

        if speed < min_speed_threshold:
            # Low speed: High noise
            noise = random.normalvariate(0, 45.0)
        else:
            # Medium/High speed: Speed-dependent noise
            sigma = max(min_sigma_deg, base_sigma / (speed + epsilon))
            noise = random.normalvariate(0, sigma)

        # Add noise and wrap around 0-360
        noisy_bearing = (true_bearing + noise) % 360.0

        # Ensure positive result if noise was very negative
        if noisy_bearing < 0:
            noisy_bearing += 360.0

        return round(noisy_bearing, 1)

    def get_random_vertices_pair(self, graph):
        source_vertex = random.randint(0, len(graph.vs) - 1)
        target_edge = None
        is_valid_pair = False
        while not is_valid_pair:
            target_edge = random.randint(0, len(graph.vs) - 1)
            if source_vertex != target_edge:
                distance = self.vertices_distance(graph, source_vertex, target_edge)
                if distance >= Config.min_route_distance:
                    is_valid_pair = True

        return source_vertex, target_edge

    def vertices_distance(self, graph, vertex_index_1, vertex_index_2):
        vertex_coords_1 = graph.vs[vertex_index_1]['geometry'].xy
        vertex_coords_1 = self.transformer.transform(vertex_coords_1[0][0], vertex_coords_1[1][0])
        vertex_coords_2 = graph.vs[vertex_index_2]['geometry'].xy
        vertex_coords_2 = self.transformer.transform(vertex_coords_2[0][0], vertex_coords_2[1][0])
        distance = ((vertex_coords_1[0] - vertex_coords_2[0]) ** 2 + (vertex_coords_1[1] - vertex_coords_2[1]) ** 2) ** 0.5
        return distance

    def project_edge_geometry(self, edge: LineString, reverse=False):
        transform_func = self.reversed_transformer.transform if reverse else self.transformer.transform
        projected_coords = [transform_func(x, y) for x, y in zip(edge.xy[0], edge.xy[1])]
        return LineString(projected_coords)



