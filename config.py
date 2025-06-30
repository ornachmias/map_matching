import os

# Output directory, just 'data' in the root dir
data_dir = os.path.realpath(os.path.join(os.path.dirname(__file__), './data'))

# Default OSM CRS, keep it
default_crs = 'epsg:4326'

# Local CRS for the map, if the map is changed it should be changed too
local_crs = 'epsg:2039'

# Map file name
map_file_name = 'israel-and-palestine-latest.osm.pbf'

# Graph file name, not really important, just for consistency
graph_file_name = 'israel-and-palestine-latest.pickle'

# Geofabrik URL for the map file, feel free to change it (but make sure to change the map_file_name too)
geofabrik_url = f'https://download.geofabrik.de/asia/{map_file_name}'

# Define the boundaries of the map to avoid loading entire territory. Change to None if you want to load the entire map.
crop_bounding_box = [34.77578823381775, 32.06442252773002, 34.81249703091649, 32.090772158773916]

# Minimum generated route distance in meters
min_route_distance = 100

# Time sampling parameters - mean and standard deviation of the normal distribution for the sampling interval (in seconds)
mean_sampling_interval = 4
sigma_sampling_interval = 3

# Maximum speed factor for the sampling, meaning how much the speed can vary from the max speed
max_speed_factor = 0.5

# Accuracy ranges for the dataset generation
accuracy_ranges = [(3.0, 15.0), (15.0, 30.0), (30.0, 60.0), (60.0, 90.0)]

# Accuracy weights for each range
accuracy_weights = [0.7, 0.25, 0.04, 0.01]
