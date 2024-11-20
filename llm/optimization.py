from docplex.mp.model import Model

def optimize_vm_placement(hosts, vms, clusters, current_usage, host_capacity, vm_demand, host_cluster):
	# Create model
	mdl = Model('vm_placement')

	# Decision variables
	x = mdl.binary_var_dict(((v, h) for v in vms for h in hosts), name='x')
	z = mdl.continuous_var(name='z')

	# Each VM must be placed exactly once
	for v in vms:
		mdl.add_constraint(mdl.sum(x[v,h] for h in hosts) == 1)

	# Resource capacity constraints
	resources = ['cpu', 'mem', 'disk']

	for h in hosts:
		for r in resources:
			mdl.add_constraint(
				mdl.sum(vm_demand[v][r] * x[v,h] for v in vms) +
				current_usage[host_cluster[h]][r] <= host_capacity[h][r]
			)

	# Min utilization constraints
	for c in clusters:
		for r in resources:
			cluster_hosts = [h for h in hosts if host_cluster[h] == c]
			total_capacity = sum(host_capacity[h][r] for h in cluster_hosts)

			mdl.add_constraint(
				(mdl.sum(vm_demand[v][r] * x[v,h]
												for v in vms
												for h in cluster_hosts) +
				current_usage[c][r]) / total_capacity >= z
			)

	# Objective
	mdl.maximize(z)

	# Solve
	solution = mdl.solve()

	return solution

# Example usage:
hosts = ['h1', 'h2', 'h3']
vms = ['vm1', 'vm2']
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
				'vm2': {'cpu': 0.3, 'mem': 0.2, 'disk': 0.2}
}

host_cluster = {
				'h1': 'c1',
				'h2': 'c1',
				'h3': 'c2'
}

solution = optimize_vm_placement(hosts, vms, clusters, current_usage,
																												host_capacity, vm_demand, host_cluster)
print(solution)
