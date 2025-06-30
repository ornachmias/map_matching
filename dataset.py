import argparse
import os
import random
import json

import igraph
from igraph import Graph
from pyproj import CRS, Transformer
from shapely import line_interpolate_point, LineString

import config
import graph_provider

input_crs = CRS.from_user_input(config.default_crs)
output_crs = CRS.from_user_input(config.local_crs)
transformer = Transformer.from_crs(input_crs, output_crs, always_xy=True)
reversed_transformer = Transformer.from_crs(output_crs, input_crs, always_xy=True)


def get_dataset(data_dir: str):
    dataset_file_path = os.path.join(data_dir, 'dataset', config.dataset_file_name)
    if not os.path.isfile(dataset_file_path):
        raise FileNotFoundError(f'Dataset file not found at {dataset_file_path}. Please build the dataset first.')
    with open(dataset_file_path, 'r') as f:
        samples = json.load(f)
    return samples


def build_dataset(data_dir: str, graph: Graph, n_routes: int):
    dataset_dir = os.path.join(data_dir, 'dataset')
    os.makedirs(dataset_dir, exist_ok=True)
    samples = get_samples(graph, n_routes)
    json.dump(samples, open(os.path.join(dataset_dir, config.dataset_file_name), 'w'), indent=4)


def get_samples(graph: Graph, n_routes: int):
    samples = []
    for route_id in range(n_routes):
        source_vertex, target_vertex = get_random_vertices_pair(graph)
        vertices_route = graph.get_shortest_path(source_vertex, target_vertex, output='vpath', mode=igraph.OUT)

        time_delta = 0
        route_samples = []
        for route_step in range(len(vertices_route)):
            step_source = vertices_route[route_step]
            step_target = vertices_route[route_step + 1] if route_step + 1 < len(vertices_route) else None
            if step_target is None:
                break

            edge_samples = get_edge_samples(graph, step_source, step_target, time_delta, route_id)
            if len(edge_samples) == 0:
                continue

            time_delta += edge_samples[-1]['timestamp'] - time_delta
            route_samples.extend(edge_samples)

        samples.append(route_samples)

    return samples

def get_edge_samples(graph, source_vertex, target_vertex, start_time, route_id):
    edge_index = graph.get_eid(source_vertex, target_vertex, directed=True)
    edge = graph.es[edge_index]
    edge_geometry = project_edge_geometry(edge['geometry'])

    total_distance = 0
    total_time = 0
    samples = []
    while total_distance < edge['length']:
        # Generate a random time interval, with minimum of 1 second
        time_interval = max([1, int(random.normalvariate(config.mean_sampling_interval, config.sigma_sampling_interval))])

        # Randomize the current speed based on the max speed
        current_speed = random.normalvariate(edge['maxspeed'], edge['maxspeed'] * config.max_speed_factor)

        # Ensure the speed is reasonable
        current_speed = max([1, current_speed])

        # Convert km/h to m/s
        current_speed = current_speed / 3.6

        # Calculate the distance covered in that time interval
        distance_covered = current_speed * time_interval

        # Check if the distance is less than the edge length
        if total_distance + distance_covered > edge['length']:
            break

        accuracy = get_random_accuracy()

        # Get the coordinates of the current location
        coords = line_interpolate_point(edge_geometry, total_distance + distance_covered)

        noise_long, noise_lat = get_coordinates_noise(accuracy)
        noisy_cords = (coords.xy[0][0] + noise_long, coords.xy[1][0] + noise_lat)
        noisy_cords = reversed_transformer.transform(noisy_cords[0], noisy_cords[1])
        coords = reversed_transformer.transform(coords.xy[0][0], coords.xy[1][0])

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
                'bearing': generate_noisy_bearing(edge['bearing'], current_speed),
                'speed': current_speed,
                'accuracy': accuracy
            }
        })

    return samples

def get_random_accuracy():
    accuracy_range = random.choices(config.accuracy_ranges, weights=config.accuracy_weights, k=1)[0]
    accuracy = random.uniform(accuracy_range[0], accuracy_range[1])
    return accuracy

def get_coordinates_noise(accuracy):
    noise_long = random.normalvariate(0, accuracy)
    noise_lat = random.normalvariate(0, accuracy)
    return noise_long, noise_lat

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

def get_random_vertices_pair(graph):
    source_vertex = random.randint(0, len(graph.vs) - 1)
    target_edge = None
    is_valid_pair = False
    while not is_valid_pair:
        target_edge = random.randint(0, len(graph.vs) - 1)
        if source_vertex != target_edge:
            distance = vertices_distance(graph, source_vertex, target_edge)
            if distance >= config.min_route_distance:
                is_valid_pair = True

    return source_vertex, target_edge

def vertices_distance(graph, vertex_index_1, vertex_index_2):
    vertex_coords_1 = graph.vs[vertex_index_1]['geometry'].xy
    vertex_coords_1 = transformer.transform(vertex_coords_1[0][0], vertex_coords_1[1][0])
    vertex_coords_2 = graph.vs[vertex_index_2]['geometry'].xy
    vertex_coords_2 = transformer.transform(vertex_coords_2[0][0], vertex_coords_2[1][0])
    distance = ((vertex_coords_1[0] - vertex_coords_2[0]) ** 2 + (vertex_coords_1[1] - vertex_coords_2[1]) ** 2) ** 0.5
    return distance

def project_edge_geometry(edge: LineString, reverse=False):
    transform_func = reversed_transformer.transform if reverse else transformer.transform
    projected_coords = [transform_func(x, y) for x, y in zip(edge.xy[0], edge.xy[1])]
    return LineString(projected_coords)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Dataset creation script')
    parser.add_argument('--data_dir', type=str, default=config.data_dir, help='Directory to store the data')
    parser.add_argument('--n_routes', type=int, default=100, help='Number of routes to generate')
    args = parser.parse_args()
    _graph = graph_provider.get_graph(args.data_dir)
    build_dataset(args.data_dir, _graph, args.n_routes)

