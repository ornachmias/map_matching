import argparse
import json

import config
import graph_provider
import dataset
from tqdm import tqdm
from algorithms.baseline import Baseline
from algorithms.topological_matching import TopologicalMatching
from algorithms.weighted_matching import WeightedMatching
from algorithms.hybrid_matching import HybridMatching

algorithm_classes = [Baseline, WeightedMatching, TopologicalMatching, HybridMatching]

def remove_consecutive_duplicates(seq):
    if not seq:
        return seq
    result = [seq[0]]
    for item in seq[1:]:
        if item != result[-1]:
            result.append(item)
    return result

def levenshtein_distance(seq1, seq2):
    seq1 = remove_consecutive_duplicates(seq1)
    seq2 = remove_consecutive_duplicates(seq2)
    m, n = len(seq1), len(seq2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq1[i - 1] == seq2[j - 1]:
                cost = 0
            else:
                cost = 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,      # deletion
                dp[i][j - 1] + 1,      # insertion
                dp[i - 1][j - 1] + cost  # substitution
            )
    return dp[m][n]

def evaluate_dataset(algorithms, samples):
    for route in tqdm(samples, desc='Evaluating routes'):
        for algorithm in algorithms:
            algorithm.reset_route()
        for sample in route:
            sample['predictions'] = {}
            sample['evaluation'] = {}
            for algorithm in algorithms:
                result = algorithm.match_sample(sample)
                sample['predictions'][algorithm.get_name()] = {
                    'edge_id': result['edge_id'],
                    'longitude': result['longitude'],
                    'latitude': result['latitude']
                }
                sample['evaluation'][algorithm.get_name()] = evaluate_sample(result, sample)
    return samples

def evaluate_route(evaluated_samples, algorithms):
    route_levenshtein = {alg.get_name(): [] for alg in algorithms}
    for route in evaluated_samples:
        actual_edges = [sample['edge_index'] for sample in route]
        for alg in algorithms:
            alg_name = alg.get_name()
            predicted_edges = [s['predictions'][alg_name]['edge_id'] for s in route]
            lev_dist = levenshtein_distance(predicted_edges, actual_edges)
            norm_sim = 1.0 - (lev_dist / max(len(actual_edges), len(predicted_edges), 1))
            route_levenshtein[alg_name].append(norm_sim)
    average_levenshtein_score = {alg: (sum(scores) / len(scores) if scores else 0) for alg, scores in route_levenshtein.items()}
    return average_levenshtein_score

def aggregate_evaluation(evaluated_samples):
    average_accuracy = aggregate_accuracy(evaluated_samples)
    average_distance = aggregate_distance(evaluated_samples)
    average_by_gps_accuracy = aggregate_by_gps_accuracy(evaluated_samples)
    average_route_score = evaluate_route(evaluated_samples, algorithm_classes)
    return {
        'average_accuracy': average_accuracy,
        'average_distance': average_distance,
        'average_by_gps_accuracy': average_by_gps_accuracy,
        'average_route_score': average_route_score
    }

def aggregate_by_gps_accuracy(evaluated_samples):
    algorithms = {}
    counts = {}
    for route in evaluated_samples:
        for sample in route:
            accuracy = sample['actual'].get('accuracy', None)
            if accuracy is None:
                continue
            accuracy_range = None
            for ar in config.accuracy_ranges:
                if ar[0] <= accuracy < ar[1]:
                    accuracy_range = f"{ar[0]}-{ar[1]}"
                    break
            if accuracy_range is None:
                continue

            if accuracy_range not in algorithms:
                algorithms[accuracy_range] = {}
                counts[accuracy_range] = 0

            for algorithm in sample['evaluation']:
                if algorithm not in algorithms[accuracy_range]:
                    algorithms[accuracy_range][algorithm] = {'distance': 0, 'classification': 0}

                algorithms[accuracy_range][algorithm]['distance'] += sample['evaluation'][algorithm]['distance']
                algorithms[accuracy_range][algorithm]['classification'] += sample['evaluation'][algorithm]['classification']
            counts[accuracy_range] += 1

    for accuracy_range in algorithms:
        for algorithm in algorithms[accuracy_range]:
            algorithms[accuracy_range][algorithm]['distance'] /= counts[accuracy_range]
            algorithms[accuracy_range][algorithm]['classification'] /= counts[accuracy_range]

    return algorithms

def aggregate_distance(evaluated_samples):
    algorithms = {a.get_name(): 0 for a in algorithm_classes}
    total_count = 0
    for route in evaluated_samples:
        for sample in route:
            for algorithm in algorithms:
                if algorithm not in algorithms:
                    algorithms[algorithm] = 0

                algorithms[algorithm] += sample['evaluation'][algorithm]['distance']
            total_count += 1
    for algorithm in algorithms:
        algorithms[algorithm] /= total_count
    return algorithms

def aggregate_accuracy(evaluated_samples):
    algorithms = {a.get_name(): 0 for a in algorithm_classes}
    total_count = 0
    for route in evaluated_samples:
        for sample in route:
            for algorithm in algorithms:
                if algorithm not in algorithms:
                    algorithms[algorithm] = 0

                algorithms[algorithm] += sample['evaluation'][algorithm]['classification']
            total_count += 1
    for algorithm in algorithms:
        algorithms[algorithm] /= total_count
    return algorithms

def evaluate_sample(prediction, sample):
    actual = sample['actual']
    pred_point = dataset.transformer.transform(prediction['longitude'], prediction['latitude'])
    actual_point = dataset.transformer.transform(actual['longitude'], actual['latitude'])
    distance = (pred_point[0] - actual_point[0]) ** 2 + (pred_point[1] - actual_point[1]) ** 2
    distance = distance ** 0.5
    classification = int(prediction['edge_id'] == sample['edge_index'])
    return {
        'distance': distance,
        'classification': classification
    }

def main():
    parser = argparse.ArgumentParser('Evaluation script')
    parser.add_argument('--data_dir', type=str, default=config.data_dir, help='Directory containing data for graph and dataset')
    args = parser.parse_args()
    _graph = graph_provider.get_graph(args.data_dir)
    _samples = dataset.get_dataset(args.data_dir)
    _algorithms = [algo_cls(_graph) for algo_cls in algorithm_classes]
    evaluated_samples = evaluate_dataset(_algorithms, _samples)
    aggregated_results = aggregate_evaluation(evaluated_samples)
    print(json.dumps(aggregated_results, indent=4))

if __name__ == '__main__':
    main()
