import json
import sys
import matplotlib.pyplot as plt
import re
import os
from collections import defaultdict 


def load_benchmark_data(json_file_path):
    with open(json_file_path, 'r') as file:
        data = json.load(file)
    return data

def extract_and_plot_data(benchmarks, output_dir):
    per_benchmark = defaultdict(lambda: {"sizes":[], "cpu_times":[]})    
    for benchmark in benchmarks:
        # Extract size from the benchmark name
        match = re.search(r'(.*)/(\d+)', benchmark['name'])
        if match:
            name = match.group(1).replace("/", "-")
            size = int(match.group(2))
            per_benchmark[name]["sizes"].append(size)
            per_benchmark[name]["cpu_times"].append(benchmark['cpu_time'])

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(10, 5))
    for name, results in per_benchmark.items():
        sizes = results["sizes"]
        cpu_times = results["cpu_times"]
        # Sort the data by size because the input might not be ordered
        sorted_data = sorted(zip(sizes, cpu_times))
        sizes_sorted, cpu_times_sorted = zip(*sorted_data)
        plt.plot(sizes_sorted, cpu_times_sorted, marker='o')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Size')
    plt.ylabel('CPU Time (ns)')
    plt.title(f'Combined Benchmark CPU Time vs Size')
    plt.grid(True)
        
    output_file_path = os.path.join(output_dir, f"combined.png")
    plt.savefig(output_file_path)
    print(f"Plot saved to {output_file_path}")
    plt.close()

    for name, results in per_benchmark.items():
        sizes = results["sizes"]
        cpu_times = results["cpu_times"]
        # Sort the data by size because the input might not be ordered
        sorted_data = sorted(zip(sizes, cpu_times))
        sizes_sorted, cpu_times_sorted = zip(*sorted_data)

        plt.figure(figsize=(10, 5))
        plt.plot(sizes_sorted, cpu_times_sorted, marker='o')
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('Size')
        plt.ylabel('CPU Time (ns)')
        plt.title(f'Benchmark {name} CPU Time vs Size')
        plt.grid(True)
        
        output_file_path = os.path.join(output_dir, f"{name}.png")
        plt.savefig(output_file_path)
        print(f"Plot saved to {output_file_path}")
        plt.close()

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python script.py <path_to_json_file> <output_directory>")
        sys.exit(1)

    json_file_path = sys.argv[1]
    output_dir = sys.argv[2]
    data = load_benchmark_data(json_file_path)
    benchmarks = data['benchmarks']
    extract_and_plot_data(benchmarks, output_dir)
