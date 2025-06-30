import os
import argparse

import requests
import igraph as ig
import numpy as np
from pyrosm import OSM

import config

def get_graph(data_dir=config.data_dir):
    """
    Load the graph from the specified directory. If it doesn't exist, download and convert it.
    """
    graph_dir = os.path.join(data_dir, 'graphs')
    os.makedirs(graph_dir, exist_ok=True)
    graph_path = os.path.join(graph_dir, config.graph_file_name)
    if os.path.isfile(graph_path):
        return ig.Graph.Read_Pickle(graph_path)

    download(graph_dir)
    map_to_graph(graph_dir)
    return ig.Graph.Read_Pickle(graph_path)

def download(graph_dir):
    """
    Download the OSM file from the geofabrik URL and save it to the output directory.
    """
    for file_name in [config.map_file_name, config.graph_file_name]:
        file_path = os.path.join(graph_dir, file_name)
        if os.path.isfile(file_path):
            print(f'File already exists at {file_path}. Skipping download.')
            return

    print(f'Downloading {config.geofabrik_url} to {graph_dir}')
    response = requests.get(config.geofabrik_url, stream=True)
    if response.status_code == 200:
        file_path = os.path.join(graph_dir, config.map_file_name)
        with open(file_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f'Downloaded {config.geofabrik_url} to {file_path}')
    else:
        print(f'Failed to download {config.geofabrik_url}. Status code: {response.status_code}')

def map_to_graph(graph_dir):
    graph_path = os.path.join(graph_dir, config.graph_file_name)
    if os.path.isfile(graph_path):
        print(f'Graph file already exists at {graph_path}. Skipping conversion.')
        return

    map_path = os.path.join(graph_dir, config.map_file_name)
    if not os.path.isfile(map_path):
        raise FileNotFoundError(f'Map file not found at {map_path}. Please download it first.')

    osm = OSM(map_path, bounding_box=config.crop_bounding_box)
    nodes, edges = osm.get_network(nodes=True, network_type='driving')
    graph = osm.to_graph(nodes, edges, graph_type='igraph')
    for edge in graph.es:
        edge['bearing'] = calculate_bearing(edge)
        if edge['maxspeed'] is None:
            edge['maxspeed'] = 35.
        else:
            edge['maxspeed'] = float(edge['maxspeed'])

    graph.write_pickle(graph_path)

def calculate_bearing(edge):
    """
    Based on OSMNX python code to calculate bearing from edge geometry.
    """
    xy = edge['geometry'].xy
    lat1, lon1, lat2, lon2 = xy[0][0], xy[1][0], xy[0][1], xy[1][1]
    lat1, lat2 = np.deg2rad(lat1), np.deg2rad(lat2)
    delta_lon = np.deg2rad(lon2 - lon1)
    y = np.sin(delta_lon) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(delta_lon)
    bearing = np.rad2deg(np.arctan2(y, x))
    return bearing % 360

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Download and process OSM data to create a graph.')
    parser.add_argument('--data_dir', type=str, default=config.data_dir, help='Directory to store the data')
    args = parser.parse_args()
    graph = get_graph(data_dir=args.data_dir)
    print(f'Graph loaded with {graph.vcount()} vertices and {graph.ecount()} edges.')
