# Map Matching Algorithms

This repository provides a framework for evaluating and comparing different map matching algorithms on GPS trajectory data. It is designed for research, benchmarking, and practical deployment of map matching solutions for road networks.

## Features
- **Multiple Algorithms:**
  - Baseline Matching
  - Weighted Matching
  - Incremental Topological Matching
  - Hybrid Matching (combines topological and weighted based on GPS accuracy)
- **Flexible Evaluation:**
  - Per-sample and per-route metrics
  - Levenshtein similarity for route sequence comparison
  - Metrics by GPS accuracy buckets
- **Visualization:**
  - Jupyter notebooks for visualizing routes, predictions, and ground truth
- **Extensible:**
  - Easily add new algorithms and metrics

## Dataset & Graph Creation
- **Automatic Dataset Creation:**
  - The repository includes scripts to automatically generate or load the GPS trajectory dataset. No manual data preparation is required.
- **Automatic Graph Acquisition:**
  - The road network graph is automatically downloaded and processed (from OpenStreetMap or other sources) if not already present. The graph is stored in the `data/graphs/` directory.

## Configuration
- All configuration options (such as accuracy buckets, data paths, and algorithm parameters) can be set in `config.py`.
- The repository is designed to work out-of-the-box; no files or parameters need to be provided by the user for basic operation.

## Repository Structure
```
algorithms/           # Map matching algorithm implementations
    baseline.py
    weighted_matching.py
    topological_matching.py
    hybrid_matching.py
config.py             # Configuration (accuracy buckets, data paths, etc.)
dataset.py            # Dataset loading and transformation utilities
graph_provider.py     # Road network graph loading and acquisition
requirements.txt      # Python dependencies
evaluation.py         # Main evaluation script
notebooks/            # Jupyter notebooks for analysis and visualization
data/
    dataset/          # GPS trajectory data (auto-generated or loaded)
    graphs/           # Road network graphs (auto-acquired)
```

## Getting Started

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Evaluation (No Setup Needed)
```bash
python evaluation.py --data_dir data
```
- The script will automatically create/load the dataset and acquire the road network graph as needed.

### 3. Visualize Results
- Use the notebooks in `notebooks/` to visualize routes, predictions, and metrics.

## Adding New Algorithms
1. Implement your algorithm in `algorithms/` as a subclass of `MatchingAlgorithm`.
2. Add it to `algorithm_classes` in `evaluation.py`.
3. (Optional) Add visualization or notebook support.

## Metrics
- **Average Accuracy:** Fraction of correct edge matches per sample
- **Average Distance:** Mean distance error per sample
- **Levenshtein Similarity:** Sequence similarity between predicted and actual edge sequences per route
- **Accuracy Buckets:** Metrics grouped by GPS accuracy

