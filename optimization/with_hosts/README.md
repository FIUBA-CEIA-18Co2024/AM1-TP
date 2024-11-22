# VM Placement Optimizer

This script optimizes the placement of virtual machines (VMs) across multiple hosts and clusters while considering resource constraints and existing placements.

## Problem Description

The optimizer solves the following problem:
- Place new VMs across available hosts while respecting resource constraints
- Take into account existing VM placements
- Consider multiple resource types (CPU, memory, disk)
- Balance resource utilization across clusters
- Maximize the minimum resource utilization across all clusters

### Constraints
- Each new VM must be placed exactly once
- Host resource capacity limits cannot be exceeded
- Existing VM placements are preserved
- Current cluster resource utilization is considered

## Requirements

- Python 3.6+
- CPLEX Optimization Studio
- Required Python packages:
  ```
  cplex==22.1.1.2
  docplex==2.28.240
  ```

## Installation

1. Install CPLEX Optimization Studio
2. Install required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

The script can be run directly:
```bash
python optimization.py
```

### Example Scenario

The example in the script demonstrates:

- 3 hosts (h1, h2, h3) across 2 clusters (c1, c2)
  - h1, h2 in cluster c1
  - h3 in cluster c2
- 2 existing VMs:
  - vm1 on h1
  - vm2 on h3
- 1 new VM (vm3) to be placed
- Resource types: CPU, memory, disk
- Each host has capacity of 1.0 (100%) for each resource
- Initial cluster utilization:
  - c1: 40% CPU, 30% memory, 50% disk
  - c2: 30% CPU, 40% memory, 20% disk

#### VM Resource Demands
- vm1: 20% CPU, 30% memory, 10% disk
- vm2: 30% CPU, 20% memory, 20% disk
- vm3: 20% CPU, 20% memory, 20% disk (new VM to place)

The optimizer will determine the optimal placement for vm3 while:
1. Respecting host capacity constraints
2. Considering existing placements
3. Maximizing minimum resource utilization across clusters

## Output

The script outputs:
1. Initial state showing existing VM placements
2. Current cluster resource utilization
3. VM resource demands
4. Optimal placement plan for new VMs
5. Resulting cluster utilization after placement
6. Achieved minimum utilization across all resources

## Customization

The example can be modified by adjusting:
- Number of hosts and clusters
- Existing VM placements
- New VMs to place
- Resource capacities and demands
- Cluster configurations
