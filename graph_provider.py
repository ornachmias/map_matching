import os

import requests
import igraph as ig
import numpy as np
from pyrosm import OSM
from config import Config

class GraphProvider:
    def __init__(self):
        self.output_dir = os.path.join(Config.output_dir, 'graphs')
        os.makedirs(self.output_dir, exist_ok=True)

    def get_graph(self):
        """
        Load the graph from the specified directory. If it doesn't exist, download and convert it.
        """
        graph_path = os.path.join(self.output_dir, Config.graph_file_name)
        if os.path.isfile(graph_path):
            return ig.Graph.Read_Pickle(graph_path)

        self.download()
        self.map_to_graph()
        return ig.Graph.Read_Pickle(graph_path)

    def download(self):
        """
        Download the OSM file from the geofabrik URL and save it to the output directory.
        """
        for file_name in [Config.map_file_name, Config.graph_file_name]:
            file_path = os.path.join(self.output_dir, file_name)
            if os.path.isfile(file_path):
                print(f'File already exists at {file_path}. Skipping download.')
                return

        print(f'Downloading {Config.geofabrik_url} to {self.output_dir}')
        response = requests.get(Config.geofabrik_url, stream=True)
        if response.status_code == 200:
            file_path = os.path.join(self.output_dir, Config.map_file_name)
            with open(file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f'Downloaded {Config.geofabrik_url} to {file_path}')
        else:
            print(f'Failed to download {Config.geofabrik_url}. Status code: {response.status_code}')

    def map_to_graph(self):
        graph_path = os.path.join(self.output_dir, Config.graph_file_name)
        if os.path.isfile(graph_path):
            print(f'Graph file already exists at {os.path.join(self.output_dir, Config.graph_file_name)}. Skipping conversion.')
            return

        map_path = os.path.join(self.output_dir, Config.map_file_name)
        if not os.path.isfile(map_path):
            raise FileNotFoundError(f'Map file not found at {os.path.join(self.output_dir, Config.map_file_name)}. Please download it first.')

        osm = OSM(map_path, bounding_box=Config.crop_bounding_box)
        nodes, edges = osm.get_network(nodes=True, network_type='driving')
        graph = osm.to_graph(nodes, edges, graph_type='igraph')
        for edge in graph.es:
            edge['bearing'] = self.calculate_bearing(edge)
            if edge['maxspeed'] is None:
                edge['maxspeed'] = 35.
            else:
                edge['maxspeed'] = float(edge['maxspeed'])

        graph.write_pickle(graph_path)

    @staticmethod
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
