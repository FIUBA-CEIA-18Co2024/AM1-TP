from docplex.mp.model import Model

def optimize_vm_placement(hosts, existing_placements, new_vms, clusters, current_usage, host_capacity, vm_demand, host_cluster):
    # Create model
    mdl = Model('vm_placement')

    # Decision variables only for new VMs
    x = mdl.binary_var_dict(((v, h) for v in new_vms for h in hosts), name='x')
    z = mdl.continuous_var(name='z')

    print("\nInitial State:")
    for vm, host in existing_placements.items():
        print(f"VM {vm} is already placed on host {host} (Cluster {host_cluster[host]})")

    # Each new VM must be placed exactly once
    for v in new_vms:
        mdl.add_constraint(mdl.sum(x[v,h] for h in hosts) == 1)

    # Resource capacity constraints
    resources = ['cpu', 'mem', 'disk']
    for h in hosts:
        for r in resources:
            # Calculate existing usage from VMs already placed on this host
            existing_usage = sum(
                vm_demand[vm][r]
                for vm, placed_host in existing_placements.items()
                if placed_host == h
            )

            # Add constraint including both existing and new VMs
            mdl.add_constraint(
                mdl.sum(vm_demand[v][r] * x[v,h] for v in new_vms) +
                existing_usage +
                current_usage[host_cluster[h]][r] <= host_capacity[h][r]
            )

    # Min utilization constraints
    for c in clusters:
        for r in resources:
            cluster_hosts = [h for h in hosts if host_cluster[h] == c]
            total_capacity = sum(host_capacity[h][r] for h in cluster_hosts)

            # Calculate existing VM usage in this cluster
            existing_cluster_usage = sum(
                vm_demand[vm][r]
                for vm, host in existing_placements.items()
                if host_cluster[host] == c
            )

            mdl.add_constraint(
                (mdl.sum(vm_demand[v][r] * x[v,h]
                    for v in new_vms
                    for h in cluster_hosts) +
                existing_cluster_usage +
                current_usage[c][r] * total_capacity) / total_capacity >= z
            )

    # Objective
    mdl.maximize(z)

    # Solve
    solution = mdl.solve()

    if solution is None:
        return None, None, None

    # Create placement plan (only for new VMs)
    placement_plan = {}
    final_utilization = solution.get_value(z)

    # Extract the placement decisions
    for v in new_vms:
        for h in hosts:
            if solution.get_value(x[v,h]) > 0.5:
                placement_plan[v] = h

    # Calculate resulting utilization for each cluster and resource
    cluster_utilization = {c: {r: 0.0 for r in resources} for c in clusters}

    # Calculate host-level utilization
    host_utilization = {h: {r: 0.0 for r in resources} for h in hosts}

    # Start with base cluster usage
    for h in hosts:
        cluster = host_cluster[h]
        for r in resources:
            host_utilization[h][r] = current_usage[cluster][r]

    # Add existing VM usage
    for vm, h in existing_placements.items():
        for r in resources:
            host_utilization[h][r] += vm_demand[vm][r]

    # Add new VM placements
    for v, h in placement_plan.items():
        for r in resources:
            host_utilization[h][r] += vm_demand[v][r]

    # Calculate cluster averages
    for c in clusters:
        cluster_hosts = [h for h in hosts if host_cluster[h] == c]
        for r in resources:
            cluster_utilization[c][r] = sum(host_utilization[h][r]
                                          for h in cluster_hosts) / len(cluster_hosts)

    return placement_plan, cluster_utilization, final_utilization

def main():
    # Example usage:
    hosts = ['h1', 'h2', 'h3']
    existing_placements = {
        'vm1': 'h1',  # vm1 is already on h1
        'vm2': 'h3'   # vm2 is already on h3
    }
    new_vms = ['vm3']  # only placing vm3
    clusters = ['c1', 'c2']

    current_usage = {
        'c1': {'cpu': 0.4, 'mem': 0.3, 'disk': 0.5},
        'c2': {'cpu': 0.3, 'mem': 0.4, 'disk': 0.2}
    }

    host_capacity = {
        'h1': {'cpu': 1.0, 'mem': 1.0, 'disk': 1.0},
        'h2': {'cpu': 1.0, 'mem': 1.0, 'disk': 1.0},
        'h3': {'cpu': 1.0, 'mem': 1.0, 'disk': 1.0}
    }

    vm_demand = {
        'vm1': {'cpu': 0.2, 'mem': 0.3, 'disk': 0.1},
        'vm2': {'cpu': 0.3, 'mem': 0.2, 'disk': 0.2},
        'vm3': {'cpu': 0.2, 'mem': 0.2, 'disk': 0.2}  # new VM to place
    }

    host_cluster = {
        'h1': 'c1',
        'h2': 'c1',
        'h3': 'c2'
    }

    print("\nCurrent Cluster State:")
    for c in clusters:
        print(f"Cluster {c}:", {r: f"{v:.1%}" for r, v in current_usage[c].items()})

    print("\nVM Resource Demands:")
    for vm, demands in vm_demand.items():
        print(f"{vm}:", {r: f"{v:.1%}" for r, v in demands.items()})

    placement_plan, cluster_utilization, final_utilization = optimize_vm_placement(
        hosts, existing_placements, new_vms, clusters, current_usage,
        host_capacity, vm_demand, host_cluster
    )

    if placement_plan:
        print("\nNew VM Placement Plan:")
        for vm, host in placement_plan.items():
            print(f"Place {vm} on host {host} (Cluster {host_cluster[host]})")

        print("\nResulting Cluster Utilization:")
        for cluster in clusters:
            print(f"\nCluster {cluster}:")
            for resource, utilization in cluster_utilization[cluster].items():
                print(f"  {resource}: {utilization:.2%}")

        print(f"\nOptimized minimum utilization across all resources: {final_utilization:.2%}")
    else:
        print("No feasible solution found")

if __name__ == "__main__":
    main()
