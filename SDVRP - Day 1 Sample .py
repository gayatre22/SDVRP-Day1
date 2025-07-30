#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import random
import math
import copy
from collections import defaultdict
import pickle # Added to load the pickle file

# --- Problem Data (from your original code) ---
DEPOT = 0
HUBS = list(range(1, 116))
NODES = [DEPOT] + HUBS
NUM_NODES = len(NODES)
NUM_LOCAL_HUBS = len(HUBS)

VEHICLE_CAPACITY = 800  # units
FIXED_VEHICLE_COST = 300  # yuan
COST_PER_KM = 6  # yuan
PENALTY_UNFULFILLED_DEMAND = 1000000 # High penalty for unfulfilled demand

# New Time Constraints for the cycles
# Specific time limits per cycle
MAX_DELIVERY_TIME_PER_VEHICLE_0_AM_TO_1_AM = 480 # minutes
MAX_DELIVERY_TIME_PER_VEHICLE_1_AM_TO_2_AM = 420 # minutes
MAX_DELIVERY_TIME_PER_VEHICLE_2_AM_TO_3_AM = 360 # minutes
MAX_DELIVERY_TIME_PER_VEHICLE_3_AM_TO_4_AM = 300 # minutes
MAX_DELIVERY_TIME_PER_VEHICLE_4_AM_TO_5_AM = 240 # minutes
MAX_DELIVERY_TIME_PER_VEHICLE_5_AM_TO_6_AM = 180 # minutes
MAX_DELIVERY_TIME_PER_VEHICLE_6_AM_TO_7_AM = 120 # minutes
MAX_DELIVERY_TIME_PER_VEHICLE_7_AM_TO_8_AM = 60 # minutes

# Create dummy distance.csv and time.csv for 116 nodes (1 depot + 115 hubs)
num_total_nodes = 116
dummy_distances = np.random.rand(num_total_nodes, num_total_nodes) * 50
np.fill_diagonal(dummy_distances, 0) # Distance to self is 0
pd.DataFrame(dummy_distances).to_csv('distance.csv', index=False, header=False)

dummy_times = np.random.rand(num_total_nodes, num_total_nodes) * 60
np.fill_diagonal(dummy_times, 0) # Time to self is 0
pd.DataFrame(dummy_times).to_csv('time.csv', index=False, header=False)

# --- Create dummy time_zero.csv for the last cycle ---
# It's based on original_time_matrix but with the DEPOT column zeroed out
dummy_times_zero_depot = np.random.rand(num_total_nodes, num_total_nodes) * 60
np.fill_diagonal(dummy_times_zero_depot, 0)
dummy_times_zero_depot[:, DEPOT] = 0.0 # Set column for DEPOT (0) to zeroes
pd.DataFrame(dummy_times_zero_depot).to_csv('time_zero.csv', index=False, header=False)


# Function to read matrices from CSV
def read_matrix_from_csv(filename):
    df = pd.read_csv(filename, header=None)
    return df.values

distances = read_matrix_from_csv('distance.csv')
original_time_matrix = read_matrix_from_csv('time.csv')
# NEW: Read time_zero_matrix
time_zero_matrix = read_matrix_from_csv('time_zero.csv')


# Node coordinates for plotting - UPDATED for 115 hubs + depot
node_coordinates = {
    0: (100.000, 83.109),  # Depot
    1: (19.699, 21.841), 2: (13.980, 16.920), 3: (51.673, 18.301), 4: (50.116, 42.524),
    5: (62.323, 37.664), 6: (70.756, 42.537), 7: (49.056, 26.630), 8: (82.360, 38.288),
    9: (47.856, 40.403), 10: (44.876, 27.375), 11: (45.700, 39.791), 12: (35.693, 24.491),
    13: (53.987, 7.820), 14: (17.640, 14.812), 15: (76.356, 40.007), 16: (39.971, 23.714),
    17: (48.416, 45.892), 18: (69.948, 44.455), 19: (0.000, 25.360), 20: (43.234, 44.760),
    21: (38.695, 63.983), 22: (34.360, 49.652), 23: (61.349, 28.422), 24: (63.241, 3.377),
    25: (58.628, 0.000), 26: (59.734, 61.066), 27: (33.518, 21.435), 28: (64.046, 9.013),
    29: (14.898, 15.291), 30: (50.650, 69.822), 31: (62.274, 33.285), 32: (65.577, 41.890),
    33: (75.507, 35.124), 34: (69.440, 37.820), 35: (67.440, 35.752), 36: (41.852, 34.377),
    37: (63.528, 46.685), 38: (78.840, 38.010), 39: (70.759, 34.382), 40: (55.180, 44.817),
    41: (61.825, 97.661), 42: (50.785, 92.922), 43: (76.875, 37.872), 44: (85.248, 71.282),
    45: (84.904, 83.396), 46: (44.386, 23.301), 47: (53.748, 37.351), 48: (52.669, 46.843),
    49: (66.718, 27.191), 50: (41.690, 31.376), 51: (57.422, 70.535), 52: (22.190, 20.998),
    53: (12.158, 10.909), 54: (52.345, 22.736), 55: (44.487, 23.528), 56: (63.813, 23.276),
    57: (68.063, 14.191), 58: (68.582, 23.157), 59: (55.678, 26.026), 60: (57.182, 33.066),
    61: (45.897, 36.170), 62: (57.030, 34.327), 63: (51.073, 26.999), 64: (63.077, 15.550),
    65: (69.910, 30.860), 66: (65.153, 19.012), 67: (58.689, 85.725), 68: (21.295, 18.918),
    69: (13.025, 16.550), 70: (59.808, 18.855), 71: (19.328, 18.304), 72: (63.815, 23.258),
    73: (16.423, 17.682), 74: (71.796, 33.410), 75: (62.491, 25.245), 76: (50.570, 45.896),
    77: (62.909, 19.050), 78: (84.779, 32.781), 79: (66.588, 15.623), 80: (54.615, 2.735),
    81: (13.721, 12.135), 82: (65.827, 23.218), 83: (13.593, 19.625), 84: (15.861, 19.925),
    85: (39.342, 32.794), 86: (57.572, 4.205), 87: (61.825, 0.394), 88: (70.715, 34.332),
    89: (70.382, 32.534), 90: (37.055, 34.915), 91: (8.806, 13.335), 92: (16.026, 14.871),
    93: (36.889, 48.800), 94: (55.879, 88.489), 95: (62.591, 7.308), 96: (21.295, 18.918),
    97: (54.202, 90.717), 98: (61.848, 19.897), 99: (57.233, 0.830), 100: (50.325, 38.494),
    101: (77.164, 42.803), 102: (70.957, 37.956), 103: (59.860, 2.252), 104: (73.958, 42.702),
    105: (58.627, 0.000), 106: (68.063, 14.191), 107: (68.986, 22.786), 108: (83.912, 81.449),
    109: (85.224, 64.752), 110: (52.228, 24.415), 111: (46.582, 49.567), 112: (56.239, 100.000),
    113: (73.964, 42.707), 114: (np.random.rand() * 100, np.random.rand() * 100), # Original random hubs kept for completeness
    115: (np.random.rand() * 100, np.random.rand() * 100) # Original random hubs kept for completeness
}

# Global variable to track active vehicles from the previous hour
previous_active_vehicles_info = {}
total_overall_cost = 0.0
hourly_results_summary = []

# NEW: Global dictionary to store vehicle usage count
vehicle_usage_counts = defaultdict(int)

# --- Helper Function for Plotting ---
def plot_routes(hour_label, routes, node_coords, title_suffix=""):
    """
    Plots the vehicle routes on a 2D plane.
    Routes are expected in the format:
    [
        {'vehicle_id': k, 'route': [depot, hub1, hub2, ..., depot], 'distance': d, 'time': t, 'deliveries': {hub: qty}},
        ...
    ]
    """
    plt.figure(figsize=(25, 20)) # Increased plot size for better readability

    safe_hour_label_str = str(hour_label).replace(':', '-').replace(' ', '_')
    # output_filename = f"vehicle_routes_hour_{safe_hour_label_str}.png" # Removed for interactive display

    # Plot nodes
    for i, (x_coord, y_coord) in node_coords.items():
        if i == DEPOT:
            plt.plot(x_coord, y_coord, 's', markersize=10, color='red', zorder=5, label='Depot (0)')
            plt.text(x_coord + 2, y_coord + 2, f'Depot (0)', fontsize=9, ha='left', va='bottom')
        else:
            plt.plot(x_coord, y_coord, 'o', markersize=8, color='blue', zorder=5, label=f'Hub ({i})' if i == HUBS[0] else "")
            plt.text(x_coord + 2, y_coord + 2, f'Hub ({i})', fontsize=9, ha='left', va='bottom')

    # Plot routes
    colors = plt.cm.get_cmap('tab20', len(routes) if routes else 1)
    for idx, vehicle_route_info in enumerate(routes):
        route = vehicle_route_info['route']
        vehicle_id = vehicle_route_info['vehicle_id']

        # Only plot routes that actually have customer visits (more than just Depot-Depot)
        if len(route) < 3:
            continue

        route_coords_x = [node_coords[node][0] for node in route]
        route_coords_y = [node_coords[node][1] for node in route]

        for i in range(len(route) - 1):
            start_x, start_y = node_coords[route[i]]
            end_x, end_y = node_coords[route[i+1]]

            plt.plot([start_x, end_x], [start_y, end_y],
                     color=colors(idx), linestyle='-', linewidth=1.5, alpha=0.8,
                     label=f'Vehicle {vehicle_id+1}' if i == 0 else "")

            # Smaller arrow heads
            dx = end_x - start_x
            dy = end_y - start_y
            plt.arrow(start_x, start_y, dx, dy,
                      head_width=1.0, head_length=1.5, fc=colors(idx), ec=colors(idx), length_includes_head=True, alpha=0.8)

    plt.title(f"Vehicle Routes for {hour_label} {title_suffix}")
    plt.xlabel("X-coordinate")
    plt.ylabel("Y-coordinate")
    plt.grid(True, linestyle='--', alpha=0.7)
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc='center left', bbox_to_anchor=(1, 0.5))

    # Add time window as text
    time_window_str = f"Time Window: ({str(hour_label).split('-')[0].strip()}, {str(hour_label).split('-')[-1].strip()})" \
                      if isinstance(hour_label, str) and '-' in str(hour_label) else \
                      f"Time Window: ({hour_label}:00, {hour_label+1}:00)"
    plt.text(0.95, 0.01, time_window_str, transform=plt.gca().transAxes,
             fontsize=12, color='red', ha='right', va='bottom',
             bbox=dict(boxstyle="round,pad=0.3", fc="yellow", ec="red", lw=1, alpha=0.8))


    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.show() # Changed from plt.savefig()
    # plt.close() # Removed plt.close() to keep plots open for display


# --- Helper Function for Printing Summary ---
def print_summary_and_plot_results(hour_label, result_data, node_coordinates_data):
    """Prints summary and calls plotting function."""
    print(f"\n--- Summary for Hour {hour_label} ---")
    print(f"  Used Vehicles: {result_data['used_vehicles_count']}")

    all_hubs_visited_this_hour = [] # Changed to list to count duplicates
    if result_data['routes']:
        for route_info in result_data['routes']:
            # Only consider routes that actually delivered something
            if route_info.get('total_delivered', 0) > 0: # Check 'total_delivered' from result
                print(f"  Vehicle {route_info['vehicle_id']+1}:")
                # Display full route path
                print(f"    Route: {' -> '.join(map(str, route_info['route']))}")
                print(f"    Distance: {route_info['distance']:.2f} km")
                print(f"    Time: {route_info['time']:.2f} minutes")

                # Ensure 'deliveries' key exists before trying to access it
                if 'deliveries' in route_info and route_info['deliveries']:
                    visited_hubs_in_route = [h for h, qty in route_info['deliveries'].items() if qty > 0]
                    print(f"    Deliveries (Hub: Quantity): {route_info['deliveries']}")
                    print(f"    Hubs visited by this vehicle: {visited_hubs_in_route}")
                    print(f"    Total Delivered by this vehicle: {sum(route_info['deliveries'].values()):.2f} units")
                    all_hubs_visited_this_hour.extend(visited_hubs_in_route) # Extend with all visits
                else:
                    print(f"    No specific deliveries recorded for this vehicle in this segment.")

    else:
        print("  No vehicle routes generated or used this hour.")

    print(f"  Total (including duplicates) hubs covered this hour: {len(all_hubs_visited_this_hour)}")
    print(f"  List of all hubs visited this hour (duplicates possible): {sorted(all_hubs_visited_this_hour)}") # Changed to reflect duplicates

    # Pass the actual hour label to plot_routes for time window display
    plot_routes(hour_label, result_data['routes'], node_coordinates_data, f"cycle")

# --- VRP Solution Representation ---
class VRPSolution:
    def __init__(self, routes_data, demands, distances_matrix, time_matrix, time_limit_per_vehicle,
                 prev_active_vehicle_info, current_hour_idx, cycle_duration_minutes, all_vehicles_ever_used_ids):

        self.demands = demands
        self.distances = distances_matrix
        self.times = time_matrix
        self.time_limit = time_limit_per_vehicle
        self.prev_active_vehicle_info = prev_active_vehicle_info
        self.current_hour_idx = current_hour_idx
        self.cycle_duration_minutes = cycle_duration_minutes
        self.all_vehicles_ever_used_ids = all_vehicles_ever_used_ids # Keep track for fixed costs

        # Deep copy routes_data to ensure independent manipulation
        self.routes = []
        for r_info in routes_data:
            route = copy.deepcopy(r_info['route'])
            deliveries = copy.deepcopy(r_info.get('deliveries', {}))

            # Ensure route starts and ends at DEPOT for consistent calculation
            if route and route[0] != DEPOT:
                route.insert(0, DEPOT)
            if route and route[-1] != DEPOT:
                route.append(DEPOT)
            if not route or (len(route) == 1 and route[0] == DEPOT):
                route = [DEPOT, DEPOT]

            # Recalculate properties based on potentially adjusted route for consistency
            dist, time = self._calculate_route_properties(route)
            self.routes.append({
                'vehicle_id': r_info.get('vehicle_id', len(self.routes)),
                'route': route,
                'distance': dist,
                'time': time,
                'current_capacity': sum(deliveries.values()), # Recalculate capacity
                'deliveries': deliveries
            })

        self.cost = self._calculate_total_cost()
        self.unfulfilled_demand = self._calculate_unfulfilled_demand()

    def _calculate_route_properties(self, route_nodes):
        """Calculates total distance and time for a given route."""
        current_dist = 0.0
        current_time = 0.0
        if len(route_nodes) < 2:
            return 0.0, 0.0
        for i in range(len(route_nodes) - 1):
            n1, n2 = route_nodes[i], route_nodes[i+1]
            current_dist += self.distances[n1][n2]
            current_time += self.times[n1][n2]
        return current_dist, current_time

    def _calculate_total_cost(self):
        total_travel_cost = 0.0
        delivered_demand_per_hub = defaultdict(float)

        total_fixed_cost = 0
        vehicles_used_this_solution_set = set()

        for r_info in self.routes:
            # Safely get current_capacity, default to 0 if not present
            if r_info.get('current_capacity', 0) > 0:
                total_travel_cost += r_info['distance'] * COST_PER_KM
                vehicles_used_this_solution_set.add(r_info['vehicle_id'])
                for hub_id, qty in r_info['deliveries'].items():
                    delivered_demand_per_hub[hub_id] += qty

        # Calculate fixed cost based on NEWLY used vehicles this hour
        # compared to previously active vehicles
        for v_id in vehicles_used_this_solution_set:
            if v_id not in self.prev_active_vehicle_info:
                total_fixed_cost += FIXED_VEHICLE_COST

        total_unfulfilled_demand_qty = 0.0
        for hub_id, demand_qty in self.demands.items():
            fulfilled_qty = delivered_demand_per_hub.get(hub_id, 0.0)
            unfulfilled = max(0.0, demand_qty - fulfilled_qty)
            total_unfulfilled_demand_qty += unfulfilled

        total_penalty_cost = total_unfulfilled_demand_qty * PENALTY_UNFULFILLED_DEMAND

        return total_travel_cost + total_fixed_cost + total_penalty_cost

    def _calculate_unfulfilled_demand(self):
        delivered_demand_per_hub = defaultdict(float)
        for r_info in self.routes:
            for hub_id, qty in r_info['deliveries'].items():
                delivered_demand_per_hub[hub_id] += qty

        unfulfilled = {h: max(0.0, self.demands[h] - delivered_demand_per_hub.get(h, 0.0)) for h in self.demands}
        return unfulfilled

    def is_feasible(self):
        # Calculate delivered_demand_per_hub within this method
        delivered_demand_per_hub = defaultdict(float)
        for r_info in self.routes:
            # Safely get current_capacity, default to 0 if not present
            if r_info.get('current_capacity', 0) > VEHICLE_CAPACITY + 1e-6:
                return False
            if self.time_limit is not None and r_info['time'] > self.time_limit + 1e-6:
                return False
            for hub_id, qty in r_info['deliveries'].items():
                delivered_demand_per_hub[hub_id] += qty # Accumulate deliveries here

        # Check for over-delivery per hub
        for hub_id, delivered_qty in delivered_demand_per_hub.items(): # Use the locally calculated variable
            if delivered_qty > self.demands.get(hub_id, 0) + 1e-6:
                return False # Delivered more than demanded, which is also infeasible if not allowed.

        # Ensure all original demands are either delivered or explicitly unfulfilled (with penalty)
        # This check is now correctly using the `delivered_demand_per_hub` calculated at the beginning of the method
        for hub_id, demand_qty in self.demands.items():
            if delivered_demand_per_hub[hub_id] > demand_qty + 1e-6:
                return False # Delivered more than demanded, which is also infeasible if not allowed.

        return True

    def get_route_customers(self, route_idx):
        return [node for node in self.routes[route_idx]['route'] if node != DEPOT]

    def get_all_customers_served(self):
        served = set()
        for r in self.routes:
            # Safely get current_capacity, default to 0 if not present
            if r.get('current_capacity', 0) > 0:
                served.update(h for h in r['deliveries'] if r['deliveries'][h] > 0)
        return served

    def copy(self):
        """Returns a deep copy of the VRPSolution."""
        return VRPSolution(
            copy.deepcopy(self.routes),
            copy.deepcopy(self.demands),
            self.distances,
            self.times,
            self.time_limit,
            copy.deepcopy(self.prev_active_vehicle_info),
            self.current_hour_idx,
            copy.deepcopy(self.cycle_duration_minutes),
            copy.deepcopy(self.all_vehicles_ever_used_ids)
        )

# --- Helper Functions for ALNS Operators ---

# Destroy Operators
class DestroyOperator:
    def __init__(self, name):
        super().__init__()
        self.name = name

    def destroy(self, solution: VRPSolution, num_to_remove):
        """
        Base method for destroy operators.
        Returns a tuple: (modified_solution, unassigned_demands)
        """
        raise NotImplementedError

class RandomRemoval(DestroyOperator):
    def __init__(self):
        super().__init__("RandomRemoval")

    def destroy(self, solution: VRPSolution, num_to_remove: int):
        modified_solution = solution.copy()
        unassigned_demands = defaultdict(float)

        # Collect all hubs that are currently being visited in the solution
        visitable_hubs = []
        for r_idx, r_info in enumerate(modified_solution.routes):
            # Safely check current_capacity
            if r_info.get('current_capacity', 0) > 0:
                for hub_id in r_info['deliveries']:
                    if r_info['deliveries'][hub_id] > 0:
                        visitable_hubs.append((r_idx, hub_id))

        if not visitable_hubs:
            return modified_solution, unassigned_demands # Nothing to remove

        random.shuffle(visitable_hubs)
        hubs_removed_count = 0

        for r_idx, hub_id in visitable_hubs:
            if hubs_removed_count >= num_to_remove:
                break

            # Retrieve the route info (might have been modified if previous hub on same route was removed)
            # Find the correct route object in modified_solution.routes by vehicle_id
            target_route = None
            for rt in modified_solution.routes:
                if rt['vehicle_id'] == modified_solution.routes[r_idx]['vehicle_id']: # Using vehicle_id to find the route
                    target_route = rt
                    break

            # Safely check for hub_id in deliveries and its quantity
            if target_route and hub_id in target_route['deliveries'] and target_route['deliveries'][hub_id] > 0:
                qty_to_remove = target_route['deliveries'][hub_id]
                unassigned_demands[hub_id] += qty_to_remove
                target_route['deliveries'][hub_id] = 0 # Mark as removed from this route
                target_route['current_capacity'] -= qty_to_remove

                # Remove hub from the route path, ensure depot returns
                target_route['route'] = [node for node in target_route['route'] if node != hub_id]

                # If only depot remains or depot-depot, ensure it's [DEPOT, DEPOT]
                if not any(n != DEPOT for n in target_route['route']):
                    target_route['route'] = [DEPOT, DEPOT]

                # Recalculate distance and time for the modified route
                target_route['distance'], target_route['time'] = \
                    modified_solution._calculate_route_properties(target_route['route'])

                hubs_removed_count += 1

        # Remove routes that now have no deliveries
        modified_solution.routes = [r for r in modified_solution.routes if r.get('current_capacity', 0) > 0 or r['route'] == [DEPOT, DEPOT]]

        # Re-calculate overall cost and unfulfilled demand for the modified solution
        modified_solution.cost = modified_solution._calculate_total_cost()
        modified_solution.unfulfilled_demand = modified_solution._calculate_unfulfilled_demand()

        return modified_solution, unassigned_demands

class WorstRemoval(DestroyOperator):
    def __init__(self):
        super().__init__("WorstRemoval")

    def destroy(self, solution: VRPSolution, num_to_remove: int):
        modified_solution = solution.copy()
        unassigned_demands = defaultdict(float)

        # Calculate cost impact for each customer
        customer_costs = {} # (vehicle_id, hub_id) -> cost_impact

        for r_idx, r_info in enumerate(modified_solution.routes):
            original_route = copy.deepcopy(r_info['route'])
            original_deliveries = copy.deepcopy(r_info['deliveries'])

            # Temporarily remove each customer and calculate cost saving
            for hub_id, qty in r_info['deliveries'].items():
                if qty <= 0: continue

                temp_route_nodes = [node for node in original_route if node != hub_id]
                if not any(n != DEPOT for n in temp_route_nodes):
                    temp_route_nodes = [DEPOT, DEPOT]

                temp_deliveries = copy.deepcopy(original_deliveries)
                temp_deliveries[hub_id] = 0

                # Create a temporary VRPSolution for comparison (only to calculate cost difference)
                # This is a bit inefficient, but ensures fixed costs are correctly handled for the comparison
                temp_routes_for_calc = []
                for rt in modified_solution.routes:
                    temp_rt_copy = copy.deepcopy(rt)
                    if temp_rt_copy['vehicle_id'] == r_info['vehicle_id']:
                        temp_rt_copy['route'] = temp_route_nodes
                        temp_rt_copy['deliveries'] = temp_deliveries
                        temp_rt_copy['current_capacity'] = sum(temp_deliveries.values())
                        # Recalculate distance and time for the temporary route directly here
                        temp_rt_copy['distance'], temp_rt_copy['time'] = solution._calculate_route_properties(temp_route_nodes)
                    temp_routes_for_calc.append(temp_rt_copy)

                temp_sol = VRPSolution(
                    temp_routes_for_calc,
                    solution.demands,
                    solution.distances,
                    solution.times,
                    solution.time_limit,
                    solution.prev_active_vehicle_info,
                    solution.current_hour_idx,
                    solution.cycle_duration_minutes,
                    solution.all_vehicles_ever_used_ids
                )

                # The cost saving is the current solution cost minus the cost without this customer
                cost_saving = solution.cost - temp_sol.cost
                customer_costs[(r_info['vehicle_id'], hub_id)] = cost_saving

        # Sort customers by their 'cost impact' in descending order (worst first)
        sorted_customers_by_cost = sorted(customer_costs.items(), key=lambda item: item[1], reverse=True)

        removed_count = 0
        for (v_id, hub_id), _ in sorted_customers_by_cost:
            if removed_count >= num_to_remove:
                break

            # Find the actual route object in modified_solution.routes
            target_route = None
            for rt in modified_solution.routes:
                if rt['vehicle_id'] == v_id:
                    target_route = rt
                    break

            # Safely check for hub_id in deliveries and its quantity
            if target_route and hub_id in target_route['deliveries'] and target_route['deliveries'][hub_id] > 0:
                qty_to_remove = target_route['deliveries'][hub_id]
                unassigned_demands[hub_id] += qty_to_remove
                target_route['deliveries'][hub_id] = 0
                target_route['current_capacity'] -= qty_to_remove

                target_route['route'] = [node for node in target_route['route'] if node != hub_id]
                if not any(n != DEPOT for n in target_route['route']):
                    target_route['route'] = [DEPOT, DEPOT]
                target_route['distance'], target_route['time'] = \
                    modified_solution._calculate_route_properties(target_route['route'])

                removed_count += 1

        # Remove routes that now have no deliveries (except DEPOT-DEPOT placeholder)
        modified_solution.routes = [r for r in modified_solution.routes if r.get('current_capacity', 0) > 0 or r['route'] == [DEPOT, DEPOT]]

        # Re-calculate overall cost and unfulfilled demand for the modified solution
        modified_solution.cost = modified_solution._calculate_total_cost()
        modified_solution.unfulfilled_demand = modified_solution._calculate_unfulfilled_demand()

        return modified_solution, unassigned_demands


# Repair Operators
class RepairOperator:
    def __init__(self, name):
        super().__init__()
        self.name = name

    def repair(self, solution: VRPSolution, unassigned_demands: dict):
        """
        Base method for repair operators.
        Returns a new VRPSolution with unassigned demands re-inserted.
        """
        raise NotImplementedError

class GreedyInsertionRepair(RepairOperator):
    def __init__(self):
        super().__init__("GreedyInsertionRepair")

    def repair(self, solution: VRPSolution, unassigned_demands_to_place: dict):
        repaired_solution = solution.copy()

        # Sort unassigned demands (e.g., largest first, or random)
        sorted_unassigned_hubs = sorted(unassigned_demands_to_place.items(), key=lambda item: item[1], reverse=True)

        current_unassigned_demands = copy.deepcopy(unassigned_demands_to_place)

        for hub_id, demand_qty_original in sorted_unassigned_hubs:
            remaining_demand_for_hub = current_unassigned_demands[hub_id]

            while remaining_demand_for_hub > 0.001:
                best_marginal_cost = float('inf')
                best_insertion_details = None # (route_idx, quantity_to_deliver, new_or_modified_route_nodes)
                chosen_vehicle_id = -1
                is_new_vehicle_insertion = False

                # 1. Try inserting into existing routes in the `repaired_solution`
                for r_idx, r_info in enumerate(repaired_solution.routes):
                    current_route_nodes = r_info['route']
                    current_load = r_info['current_capacity']

                    qty_possible_by_capacity = VEHICLE_CAPACITY - current_load
                    qty_to_try_in_this_vehicle = min(remaining_demand_for_hub, qty_possible_by_capacity)

                    if qty_to_try_in_this_vehicle <= 0.001:
                        continue

                    temp_route_for_calc = None

                    if hub_id in current_route_nodes:
                        temp_route_for_calc = current_route_nodes # No path change
                        marginal_cost_for_insertion = 0.0 # No change in travel cost
                    else:
                        temp_marginal_cost_for_path = float('inf')
                        temp_best_path = None

                        # Find best insertion point in current route for this hub
                        # Special handling for routes that are just [DEPOT, DEPOT]
                        if len(current_route_nodes) == 2 and current_route_nodes == [DEPOT, DEPOT]:
                            # Inserting into an empty route
                            temp_path_for_time_check = [DEPOT, hub_id, DEPOT]
                            _, temp_time = solution._calculate_route_properties(temp_path_for_time_check)

                            if solution.time_limit is not None and temp_time > solution.time_limit + 1e-6:
                                continue # Infeasible by time limit

                            marginal_path_cost = solution._calculate_route_properties(temp_path_for_time_check)[0] * COST_PER_KM

                            temp_marginal_cost_for_path = marginal_path_cost
                            temp_best_path = temp_path_for_time_check
                        else: # For non-empty routes, insert between any two existing nodes
                            for i in range(len(current_route_nodes) - 1):
                                node_before = current_route_nodes[i]
                                node_after = current_route_nodes[i+1]

                                # Temporarily form the new path to check time feasibility
                                temp_path_for_time_check = current_route_nodes[:i+1] + [hub_id] + current_route_nodes[i+1:]
                                _, temp_time = solution._calculate_route_properties(temp_path_for_time_check)

                                if solution.time_limit is not None and temp_time > solution.time_limit + 1e-6:
                                    continue # Infeasible by time limit

                                # Calculate marginal travel cost
                                original_dist = solution._calculate_route_properties(current_route_nodes)[0]
                                new_dist_temp = solution._calculate_route_properties(temp_path_for_time_check)[0]
                                marginal_path_cost = (new_dist_temp - original_dist) * COST_PER_KM

                                if marginal_path_cost < temp_marginal_cost_for_path:
                                    temp_marginal_cost_for_path = marginal_path_cost
                                    temp_best_path = temp_path_for_time_check

                        if temp_best_path is not None:
                            temp_route_for_calc = temp_best_path
                            marginal_cost_for_insertion = temp_marginal_cost_for_path
                        else: # No feasible insertion point found in this route
                            continue

                    # Evaluate total marginal cost including fixed costs for this option
                    if marginal_cost_for_insertion < best_marginal_cost:
                        best_marginal_cost = marginal_cost_for_insertion
                        best_insertion_details = (r_idx, qty_to_try_in_this_vehicle, temp_route_for_calc)
                        chosen_vehicle_id = r_info['vehicle_id']
                        is_new_vehicle_insertion = False # Not a new vehicle for costing


                # 2. Try creating a *brand new vehicle* or using a *previously active vehicle*
                # Prioritize available prev-active vehicles
                available_prev_vehicles_for_new_route = sorted([
                    v_id for v_id, info in solution.prev_active_vehicle_info.items()
                    if info['available_at_hour'] <= solution.current_hour_idx and v_id not in [r['vehicle_id'] for r in repaired_solution.routes] # not currently in use
                ])

                candidate_vehicles_for_new_route_options = []
                for v_id in available_prev_vehicles_for_new_route:
                    start_node = solution.prev_active_vehicle_info[v_id]['last_node']
                    candidate_vehicles_for_new_route_options.append((v_id, start_node, False)) # (vehicle_id, start_node, is_new_vehicle_for_costing)

                # Add option for a brand new vehicle
                next_new_vehicle_id = 0
                if solution.all_vehicles_ever_used_ids:
                    next_new_vehicle_id = max(solution.all_vehicles_ever_used_ids) + 1
                candidate_vehicles_for_new_route_options.append((next_new_vehicle_id, DEPOT, True)) # Always starts at DEPOT for a truly new route

                for v_id_candidate, start_node_candidate, is_new_cost in candidate_vehicles_for_new_route_options:
                    qty_to_try = min(remaining_demand_for_hub, VEHICLE_CAPACITY)
                    if qty_to_try <= 0.001: continue

                    new_route_nodes = [start_node_candidate, hub_id, DEPOT] if start_node_candidate != DEPOT else [DEPOT, hub_id, DEPOT]

                    new_route_dist, new_route_time = solution._calculate_route_properties(new_route_nodes)

                    if solution.time_limit is not None and new_route_time > solution.time_limit + 1e-6:
                        continue

                    # Marginal cost for this new route (includes fixed cost if it's a truly new vehicle)
                    cost_of_this_option = FIXED_VEHICLE_COST + new_route_dist * COST_PER_KM
                    if is_new_cost:
                        cost_of_this_option += FIXED_VEHICLE_COST # Only add fixed cost ONCE per vehicle

                    if cost_of_this_option < best_marginal_cost:
                        best_marginal_cost = cost_of_this_option
                        # Using -1 for r_idx_placeholder means it's a new route
                        best_insertion_details = (-1, qty_to_try, new_route_nodes)
                        chosen_vehicle_id = v_id_candidate
                        is_new_vehicle_insertion = is_new_cost


                # Apply the best insertion found
                if best_insertion_details:
                    r_idx_or_new_flag, quantity_delivered_now, new_or_modified_route_nodes = best_insertion_details

                    target_route_info = None
                    # Find the route for the chosen_vehicle_id among the current routes being built
                    for rt_idx, rt_info in enumerate(repaired_solution.routes):
                        if rt_info['vehicle_id'] == chosen_vehicle_id:
                            target_route_info = rt_info
                            r_idx = rt_idx # Store the actual index if found
                            break

                    if target_route_info: # Update an existing route for this vehicle in `repaired_solution.routes`
                        # If hub already visited, don't change route path, just add deliveries
                        if hub_id not in target_route_info['route']: # Check if the hub is truly a new stop in the path
                            target_route_info['route'] = new_or_modified_route_nodes # Update path
                            target_route_info['distance'], target_route_info['time'] = \
                                repaired_solution._calculate_route_properties(target_route_info['route'])

                        target_route_info['deliveries'][hub_id] = target_route_info['deliveries'].get(hub_id, 0) + quantity_delivered_now
                        target_route_info['current_capacity'] = sum(target_route_info['deliveries'].values())

                    else: # Create a new route for this vehicle (either prev-active or brand new)
                        repaired_solution.routes.append({
                            'vehicle_id': chosen_vehicle_id,
                            'route': new_or_modified_route_nodes,
                            'deliveries': {hub_id: quantity_delivered_now},
                            'distance': 0, # Placeholder, will be recalculated
                            'time': 0, # Placeholder
                            'current_capacity': quantity_delivered_now
                        })
                        repaired_solution.routes[-1]['distance'], repaired_solution.routes[-1]['time'] = \
                            repaired_solution._calculate_route_properties(repaired_solution.routes[-1]['route'])
                        repaired_solution.all_vehicles_ever_used_ids.add(chosen_vehicle_id)

                    remaining_demand_for_hub -= quantity_delivered_now
                    current_unassigned_demands[hub_id] = remaining_demand_for_hub
                    if remaining_demand_for_hub < 0.001: remaining_demand_for_hub = 0.0 # avoid negative float precision issues
                else:
                    # No feasible insertion found for this segment of demand for the current hub.
                    # This segment of demand remains unassigned for now.
                    break # Move to next hub

        # Clean up empty routes (unless they are [DEPOT, DEPOT] for already active vehicles)
        initial_vehicles_in_repair_sol = {r['vehicle_id'] for r in solution.routes}
        repaired_solution.routes = [r for r in repaired_solution.routes if r.get('current_capacity', 0) > 0 or (r['vehicle_id'] in initial_vehicles_in_repair_sol and len(r['route']) <= 2)]

        # Ensure depot returns for all vehicles, and re-calculate accurate properties
        for r_info in repaired_solution.routes:
            # Check if current_capacity exists, if not, assume it's 0 for logic below
            if r_info.get('current_capacity', 0) > 0: # Active route
                # Ensure it starts and ends at DEPOT if it's not a prev-active vehicle that might be mid-route
                if r_info['route'][0] != DEPOT:
                    r_info['route'].insert(0, DEPOT)
                if r_info['route'][-1] != DEPOT:
                    r_info['route'].append(DEPOT)

            # Remove duplicate consecutive nodes
            clean_route = [r_info['route'][0]]
            for i in range(1, len(r_info['route'])):
                if r_info['route'][i] != r_info['route'][i-1]:
                    clean_route.append(r_info['route'][i])
            r_info['route'] = clean_route

            r_info['distance'], r_info['time'] = repaired_solution._calculate_route_properties(r_info['route'])


        # Re-calculate overall cost and unfulfilled demand for the repaired solution
        repaired_solution.cost = repaired_solution._calculate_total_cost()
        repaired_solution.unfulfilled_demand = repaired_solution._calculate_unfulfilled_demand()

        return repaired_solution


class ALNS:
    def __init__(self, initial_solution: VRPSolution, destroy_operators: list, repair_operators: list,
                 max_iterations: int, initial_temperature: float, cooling_rate: float,
                 reaction_factor: float, segment_length: int, score_params: dict):
        self.current_solution = initial_solution
        self.best_solution = initial_solution.copy()

        self.destroy_operators = destroy_operators
        self.repair_operators = repair_operators

        self.max_iterations = max_iterations
        self.temperature = initial_temperature
        self.cooling_rate = cooling_rate
        self.reaction_factor = reaction_factor
        self.segment_length = segment_length
        self.score_params = score_params # e.g., {'s1': 10, 's2': 5, 's3': 1, 's4': 0}

        # Initialize operator weights and scores
        self.operator_weights = defaultdict(lambda: 1.0)
        self.operator_scores = defaultdict(float)
        self.operator_usages = defaultdict(int)

        for op in self.destroy_operators + self.repair_operators:
            self.operator_weights[op.name] = 1.0
            self.operator_scores[op.name] = 0.0
            self.operator_usages[op.name] = 0

    def select_operator(self, operators):
        """Selects an operator based on its weight using roulette wheel selection."""
        total_weight = sum(self.operator_weights[op.name] for op in operators)
        if total_weight == 0: # Fallback if all weights are zero (shouldn't happen with init to 1.0)
            return random.choice(operators)

        pick = random.uniform(0, total_weight)
        current = 0
        for op in operators:
            current += self.operator_weights[op.name]
            if current > pick:
                return op
        return random.choice(operators) # Fallback

    def update_weights(self):
        for op in self.destroy_operators + self.repair_operators:
            if self.operator_usages[op.name] > 0:
                # Average score for this segment
                avg_score = self.operator_scores[op.name] / self.operator_usages[op.name]
                # Update weight
                self.operator_weights[op.name] = (
                    self.operator_weights[op.name] * (1.0 - self.reaction_factor) +
                    avg_score * self.reaction_factor
                )
            # Reset scores and usages for the next segment
            self.operator_scores[op.name] = 0.0
            self.operator_usages[op.name] = 0

    def solve(self):
        print(f"  ALNS: Starting with initial cost: {self.current_solution.cost:.2f}")
        for iteration in range(self.max_iterations):
            if iteration % self.segment_length == 0 and iteration > 0:
                self.update_weights()

            # Select destroy and repair operators
            destroy_op = self.select_operator(self.destroy_operators)
            repair_op = self.select_operator(self.repair_operators)

            self.operator_usages[destroy_op.name] += 1
            self.operator_usages[repair_op.name] += 1

            # Determine number of customers to remove (e.g., between 5% and 20% of total served)
            num_customers_to_destroy = len(self.current_solution.get_all_customers_served())
            if num_customers_to_destroy > 0:
                remove_count = random.randint(1, min(max(1, int(num_customers_to_destroy * 0.1)), num_customers_to_destroy))
            else:
                remove_count = 0 # If no customers, no removal needed

            # Destroy
            destroyed_solution, unassigned_demands = destroy_op.destroy(self.current_solution, remove_count)

            # Repair
            new_solution = repair_op.repair(destroyed_solution, unassigned_demands)

            # Accept or Reject
            cost_current = self.current_solution.cost
            cost_new = new_solution.cost

            if not new_solution.is_feasible():
                # Penalize highly if infeasible
                self.operator_scores[destroy_op.name] += self.score_params.get('s4', 0)
                self.operator_scores[repair_op.name] += self.score_params.get('s4', 0)
                continue # Do not accept infeasible solution

            if cost_new < self.best_solution.cost:
                # New global best
                self.best_solution = new_solution.copy()
                self.current_solution = new_solution.copy()
                self.operator_scores[destroy_op.name] += self.score_params.get('s1', 30)
                self.operator_scores[repair_op.name] += self.score_params.get('s1', 30)
                print(f"    Iter {iteration+1}: New best found! Cost: {self.best_solution.cost:.2f}")
            elif cost_new < cost_current:
                # Better than current, but not global best
                self.current_solution = new_solution.copy()
                self.operator_scores[destroy_op.name] += self.score_params.get('s2', 20)
                self.operator_scores[repair_op.name] += self.score_params.get('s2', 20)
            else:
                # Worse than current, apply simulated annealing acceptance
                delta_cost = cost_new - cost_current
                acceptance_prob = math.exp(-delta_cost / self.temperature)
                if random.random() < acceptance_prob:
                    self.current_solution = new_solution.copy()
                    self.operator_scores[destroy_op.name] += self.score_params.get('s3', 10)
                    self.operator_scores[repair_op.name] += self.score_params.get('s3', 10)
                else:
                    self.operator_scores[destroy_op.name] += self.score_params.get('s4', 0)
                    self.operator_scores[repair_op.name] += self.score_params.get('s4', 0)

            # Annealing schedule
            self.temperature *= self.cooling_rate
            if self.temperature < 1e-6: # Prevent temperature from becoming too small
                self.temperature = 1e-6

        print(f"  ALNS: Finished. Best cost found: {self.best_solution.cost:.2f}")
        return self.best_solution

# --- Clarke-Wright Savings Algorithm for Initial Solution ---

def clarke_wright_savings(demands_for_hour, distances_matrix, time_matrix, time_limit_per_vehicle,
                            prev_active_vehicles_info_param, current_hour_idx, all_vehicles_ever_used_ids_ref):
    """
    Generates an initial solution using the Clarke-Wright Savings Algorithm.
    Prioritizes merging existing routes rather than creating new vehicles initially,
    and then incorporates demands that couldn't be merged.
    """
    routes_data = []
    active_hubs = [h for h, d in demands_for_hour.items() if d > 0.001]

    if not active_hubs:
        return [], all_vehicles_ever_used_ids_ref

    # Initialize remaining_demands_to_assign here
    remaining_demands_to_assign = demands_for_hour.copy()

    # Create a dummy VRPSolution object (or pass minimal data to calculate route properties)
    # This is a bit of a workaround to reuse the _calculate_route_properties logic
    # without creating a full, complex VRPSolution just for initial route calculations.
    # We pass the matrices and time limit for its internal calculations.
    temp_calc_sol = VRPSolution([], {}, distances_matrix, time_matrix, time_limit_per_vehicle,
                                 prev_active_vehicles_info_param, current_hour_idx, 0, set())

    # Assign demands, preferring existing vehicles first
    current_vehicle_routes = {} # {vehicle_id: {'route': [...], 'deliveries': {...}, 'current_capacity': float}}

    # Initialize current_vehicle_routes with previously active vehicles, starting from their last node
    for v_id, info in prev_active_vehicles_info_param.items():
        if info['available_at_hour'] <= current_hour_idx:
            initial_route = [info['last_node'], DEPOT] # Default: from last node back to depot
            initial_dist, initial_time = temp_calc_sol._calculate_route_properties(initial_route)
            current_vehicle_routes[v_id] = {
                'vehicle_id': v_id,
                'route': initial_route,
                'deliveries': defaultdict(float),
                'current_capacity': 0.0,
                'distance': initial_dist, # Initial distance for just returning to depot if no new tasks
                'time': initial_time # Initial time for just returning to depot
            }
            all_vehicles_ever_used_ids_ref.add(v_id)


    # Greedy assignment of demands (can be part of CW or a separate initial heuristic)
    for hub_id in sorted(active_hubs, key=lambda h: demands_for_hour[h], reverse=True):
        demand_needed = remaining_demands_to_assign[hub_id]

        while demand_needed > 0.001:
            best_vehicle_for_hub = None
            best_qty_to_assign = 0
            best_cost_impact = float('inf')
            best_path_for_vehicle = None

            # Try existing routes
            for v_id, route_info in current_vehicle_routes.items():
                current_load = route_info['current_capacity']
                qty_possible_by_capacity = VEHICLE_CAPACITY - current_load
                qty_to_try_in_this_vehicle = min(demand_needed, qty_possible_by_capacity)

                if qty_to_try_in_this_vehicle <= 0.001:
                    continue

                temp_route_nodes = list(route_info['route']) # Make a mutable copy
                prospective_total_time = route_info['time']
                marginal_cost_for_path_change = 0.0

                if hub_id not in temp_route_nodes:
                    # Find best insertion point for the hub if it's new to this route

                    temp_best_insertion_cost_path = float('inf')
                    temp_insertion_path = None

                    # Iterate through all possible insertion points. Ensure not to insert within DEPOT-DEPOT segment
                    # unless it's a non-empty route.
                    # If route is [DEPOT, DEPOT], new insertions happen between DEPOT and DEPOT
                    if len(temp_route_nodes) == 2 and temp_route_nodes == [DEPOT, DEPOT]:
                        # Special case for empty route, insert between depot and depot
                        node_before = DEPOT
                        node_after = DEPOT

                        cost_added = temp_calc_sol.distances[node_before][hub_id] + temp_calc_sol.distances[hub_id][node_after]
                        time_added = temp_calc_sol.times[node_before][hub_id] + temp_calc_sol.times[hub_id][node_after]

                        prospective_total_time_candidate = time_added

                        if time_limit_per_vehicle is not None and prospective_total_time_candidate > time_limit_per_vehicle + 1e-6:
                            continue # Infeasible by time limit

                        if cost_added * COST_PER_KM < temp_best_insertion_cost_path:
                            temp_best_insertion_cost_path = cost_added * COST_PER_KM
                            temp_insertion_path = [DEPOT, hub_id, DEPOT]
                            prospective_total_time = prospective_total_time_candidate
                            marginal_cost_for_path_change = temp_best_insertion_cost_path
                    else: # For non-empty routes, insert between any two existing nodes
                        for i in range(len(temp_route_nodes) - 1):
                            node_before = temp_route_nodes[i]
                            node_after = temp_route_nodes[i+1]

                            cost_added = temp_calc_sol.distances[node_before][hub_id] + temp_calc_sol.distances[hub_id][node_after] - temp_calc_sol.distances[node_before][node_after]
                            time_added = temp_calc_sol.times[node_before][hub_id] + temp_calc_sol.times[hub_id][node_after] - temp_calc_sol.times[node_before][node_after]

                            prospective_total_time_candidate = route_info['time'] + time_added

                            if time_limit_per_vehicle is not None and prospective_total_time_candidate > time_limit_per_vehicle + 1e-6:
                                continue # Infeasible by time limit

                            if cost_added * COST_PER_KM < temp_best_insertion_cost_path:
                                temp_best_insertion_cost_path = cost_added * COST_PER_KM
                                temp_insertion_path = temp_route_nodes[:i+1] + [hub_id] + temp_route_nodes[i+1:]
                                prospective_total_time = prospective_total_time_candidate # Update time for this best path
                                marginal_cost_for_path_change = temp_best_insertion_cost_path

                    if temp_insertion_path:
                        temp_route_nodes = temp_insertion_path
                    else: # No feasible insertion point found in this route
                        continue # Skip this vehicle, try another
                # Else (hub_id is already in temp_route_nodes), marginal_cost_for_path_change remains 0.0 and temp_route_nodes remains original

                cost_impact = marginal_cost_for_path_change # Only travel cost for existing route modification

                if cost_impact < best_cost_impact:
                    best_cost_impact = cost_impact
                    best_vehicle_for_hub = v_id
                    best_qty_to_assign = qty_to_try_in_this_vehicle
                    best_path_for_vehicle = temp_route_nodes # Store the new path

            # If no existing route could take the demand (or part of it), consider a new vehicle
            if best_vehicle_for_hub is None:
                qty_to_assign = min(demand_needed, VEHICLE_CAPACITY)
                if qty_to_assign <= 0.001:
                    break # Cannot assign this demand even to a new vehicle

                next_new_v_id = 0
                if all_vehicles_ever_used_ids_ref:
                    next_new_v_id = max(all_vehicles_ever_used_ids_ref) + 1

                new_route_nodes_for_new_vehicle = [DEPOT, hub_id, DEPOT]
                new_dist, new_time = temp_calc_sol._calculate_route_properties(new_route_nodes_for_new_vehicle)

                if time_limit_per_vehicle is not None and new_time > time_limit_per_vehicle + 1e-6:
                    # This new single-customer route is already infeasible by time
                    demand_needed = 0 # Mark this demand as unfulfillable for this iteration
                    break # Stop trying to place this specific demand piece

                cost_of_this_option = FIXED_VEHICLE_COST + new_dist * COST_PER_KM

                if cost_of_this_option < best_cost_impact:
                    best_cost_impact = cost_of_this_option
                    best_vehicle_for_hub = next_new_v_id
                    best_qty_to_assign = qty_to_assign
                    best_path_for_vehicle = new_route_nodes_for_new_vehicle

            # Apply the best assignment found
            if best_vehicle_for_hub is not None:
                if best_vehicle_for_hub not in current_vehicle_routes:
                    current_vehicle_routes[best_vehicle_for_hub] = {
                        'vehicle_id': best_vehicle_for_hub,
                        'route': best_path_for_vehicle,
                        'deliveries': defaultdict(float),
                        'current_capacity': 0.0,
                        'distance': 0.0,
                        'time': 0.0
                    }
                    all_vehicles_ever_used_ids_ref.add(best_vehicle_for_hub)

                target_route_info = current_vehicle_routes[best_vehicle_for_hub]
                target_route_info['deliveries'][hub_id] += best_qty_to_assign
                target_route_info['current_capacity'] = sum(target_route_info['deliveries'].values())
                target_route_info['route'] = best_path_for_vehicle # Update path (only if hub was new to route)
                target_route_info['distance'], target_route_info['time'] = \
                    temp_calc_sol._calculate_route_properties(target_route_info['route'])

                remaining_demands_to_assign[hub_id] -= best_qty_to_assign
                demand_needed = remaining_demands_to_assign[hub_id]
                if demand_needed < 0.001: demand_needed = 0.0
            else:
                # If no feasible assignment found (even with new vehicle), this piece of demand remains unassigned
                # This should ideally only happen if a single customer's demand is larger than vehicle capacity
                # or if no route can be formed within time limits.
                break # Move to next hub, can't satisfy this one right now

    # Convert current_vehicle_routes to a list for return
    routes_data = list(current_vehicle_routes.values())

    # Final cleanup of routes for return
    final_routes_ordered = []
    routes_data.sort(key=lambda r: r['vehicle_id'])
    for r_info in routes_data:
        # Ensure unique nodes in route (excluding depot loops unless meaningful for path)
        processed_route = []
        if r_info['route']:
            processed_route.append(r_info['route'][0])
            for i in range(1, len(r_info['route'])):
                if r_info['route'][i] != r_info['route'][i-1]:
                    processed_route.append(r_info['route'][i])

        # Ensure start and end at depot for plotting and consistency
        if DEPOT not in processed_route:
            processed_route.insert(0, DEPOT)
            processed_route.append(DEPOT)
        elif processed_route[0] != DEPOT:
            processed_route.insert(0, DEPOT)
        if processed_route[-1] != DEPOT:
            processed_route.append(DEPOT)

        r_info['route'] = processed_route
        r_info['distance'], r_info['time'] = temp_calc_sol._calculate_route_properties(r_info['route'])
        final_routes_ordered.append(r_info)

    return final_routes_ordered, all_vehicles_ever_used_ids_ref


# --- Main Solver Function using ALNS ---
all_vehicles_ever_used_ids = set() # Global set to track all vehicle IDs

def solve_vrp_with_alns(hour_info, distances_matrix, time_matrix_to_use, max_delivery_time_limit=None,
                            prev_active_vehicles_info_param=None, is_final_push=False):
    """
    Solves the VRP for a given hour using ALNS.
    Handles vehicle reuse logic and fixed costs.
    """
    global all_vehicles_ever_used_ids, vehicle_usage_counts

    current_hour_label = hour_info['hour']
    current_demands = hour_info['demands']
    current_hour_idx = hour_info.get('hour_index', -1)

    # Determine current cycle duration for availability calculation
    if isinstance(current_hour_label, int):
        cycle_duration_minutes = 60
    elif current_hour_label == "6:00-7:00 AM":
        cycle_duration_minutes = 60
    elif current_hour_label == "7:00-8:00 AM":
        cycle_duration_minutes = 60
    else:
        cycle_duration_minutes = 60

    actual_demands = {h: d for h, d in current_demands.items() if d > 0.001}

    if not actual_demands:
        print(f"No active demands for {current_hour_label}. Returning empty solution.")
        return {
            'status': 'NO_DEMAND',
            'objective_value': 0.0,
            'used_vehicles_count': 0,
            'routes': [],
            'deliveries': [],
            'active_vehicles_for_next_hour': {},
            'hour': current_hour_label,
            'unfulfilled_demand_per_hub': {j: 0.0 for j in HUBS}
        }

    print(f"\n--- ALNS Solver: Starting for {current_hour_label} ---")

    # Step 1: Generate initial solution using Clarke-Wright Savings
    initial_routes_data_for_alns, _ = clarke_wright_savings(
        actual_demands, distances_matrix, time_matrix_to_use, max_delivery_time_limit,
        prev_active_vehicles_info_param, current_hour_idx, all_vehicles_ever_used_ids # Pass global set
    )

    initial_solution_alns = VRPSolution(
        initial_routes_data_for_alns,
        actual_demands,
        distances_matrix,
        time_matrix_to_use,
        max_delivery_time_limit,
        prev_active_vehicles_info_param,
        current_hour_idx,
        cycle_duration_minutes,
        all_vehicles_ever_used_ids # Pass the potentially updated global set
    )

    if not initial_solution_alns.is_feasible():
        print(f"  WARNING: Initial solution for {current_hour_label} is infeasible. ALNS might struggle.")

    print(f"  Initial Solution Cost: {initial_solution_alns.cost:.2f}, Used Vehicles: {len(initial_solution_alns.routes)}")

    # Step 2: Configure and run ALNS
    destroy_ops = [RandomRemoval(), WorstRemoval()]
    repair_ops = [GreedyInsertionRepair()] # Add more repair operators if desired

    alns_params = {
        'max_iterations': 500, # Number of ALNS iterations
        'initial_temperature': 1000.0, # Initial temperature for SA
        'cooling_rate': 0.99, # Cooling rate for SA
        'reaction_factor': 0.1, # How quickly weights adapt
        'segment_length': 50, # How often weights are updated
        'score_params': {'s1': 30, 's2': 20, 's3': 10, 's4': 0} # Scores for operator performance
    }

    alns_solver = ALNS(initial_solution_alns, destroy_ops, repair_ops, **alns_params)
    final_alns_solution = alns_solver.solve()

    if not final_alns_solution.is_feasible():
        print(f"  WARNING: Final ALNS solution for {current_hour_label} is infeasible after search.")

    # Process and return results
    used_vehicles_count = len([r for r in final_alns_solution.routes if r.get('current_capacity', 0) > 0])

    final_routes_formatted = []
    final_deliveries_formatted = []

    current_active_vehicle_ids = set() # Track vehicles active THIS hour
    for r_info in final_alns_solution.routes:
        if r_info.get('current_capacity', 0) > 0: # Only include routes that actually delivered something
            final_routes_formatted.append({
                'vehicle_id': r_info['vehicle_id'],
                'route': r_info['route'],
                'distance': r_info['distance'],
                'time': r_info['time'],
                'deliveries': r_info['deliveries'],
                'total_delivered': r_info['current_capacity']
            })
            final_deliveries_formatted.append({
                'vehicle_id': r_info['vehicle_id'],
                'deliveries': r_info['deliveries'],
                'total_delivered': r_info['current_capacity']
            })
            current_active_vehicle_ids.add(r_info['vehicle_id'])

    # Update vehicle usage counts
    for v_id in current_active_vehicle_ids:
        vehicle_usage_counts[v_id] += 1

    # Calculate next hour's active vehicle info based on this hour's results
    next_hour_active_vehicles_info = {}
    if not is_final_push:
        for r_info in final_alns_solution.routes:
            if r_info.get('current_capacity', 0) > 0 and len(r_info['route']) > 1:
                vehicle_id = r_info['vehicle_id']
                total_vehicle_time = r_info['time']

                trip_end_absolute_minutes = (current_hour_idx * 60) + total_vehicle_time
                next_available_start_hour_absolute = math.ceil(trip_end_absolute_minutes / 60.0)

                next_hour_active_vehicles_info[vehicle_id] = {
                    'last_node': r_info['route'][-2] if len(r_info['route']) >= 2 and r_info['route'][-1] == DEPOT else r_info['route'][-1],
                    'available_at_hour': next_available_start_hour_absolute
                }

    print(f"--- ALNS Solver: Finished for {current_hour_label} ---")

    return {
        'status': 'ALNS_COMPLETED',
        'objective_value': final_alns_solution.cost,
        'used_vehicles_count': used_vehicles_count,
        'routes': final_routes_formatted,
        'deliveries': final_deliveries_formatted,
        'active_vehicles_for_next_hour': next_hour_active_vehicles_info,
        'hour': current_hour_label,
        'unfulfilled_demand_per_hub': final_alns_solution.unfulfilled_demand
    }


# --- Main Simulation Loop for Hourly Demands (12 AM to 6 AM) ---

# START OF MODIFIED DEMAND INPUT BLOCK
# Provided demand data from previous turn, simulating hourly_data.pkl content
hub_ids_str_list = ['LH_0', 'LH_1', 'LH_10', 'LH_101', 'LH_102', 'LH_103', 'LH_105', 'LH_106', 'LH_107', 'LH_108', 'LH_11', 'LH_110', 'LH_111', 'LH_112', 'LH_113', 'LH_12', 'LH_13', 'LH_14', 'LH_15', 'LH_16', 'LH_17', 'LH_18', 'LH_19', 'LH_2', 'LH_20', 'LH_21', 'LH_22', 'LH_23', 'LH_24', 'LH_25', 'LH_27', 'LH_28', 'LH_29', 'LH_30', 'LH_32', 'LH_34', 'LH_35', 'LH_36', 'LH_37', 'LH_38', 'LH_4', 'LH_40', 'LH_41', 'LH_44', 'LH_46', 'LH_47', 'LH_48', 'LH_5', 'LH_50', 'LH_52', 'LH_53', 'LH_54', 'LH_55', 'LH_56', 'LH_57', 'LH_59', 'LH_6', 'LH_60', 'LH_63', 'LH_66', 'LH_67', 'LH_68', 'LH_69', 'LH_7', 'LH_70', 'LH_73', 'LH_74', 'LH_75', 'LH_76', 'LH_77', 'LH_79', 'LH_8', 'LH_80', 'LH_81', 'LH_82', 'LH_84', 'LH_85', 'LH_86', 'LH_87', 'LH_88', 'LH_9', 'LH_93', 'LH_94', 'LH_96', 'LH_98']
demand_matrix_raw = [[0, np.int64(21), np.int64(361), np.int64(843), np.int64(1209), np.int64(848), 0], [0, 0, np.int64(110), np.int64(169), 0, np.int64(205), 0], [0, 0, 0, np.int64(35), np.int64(27), np.int64(38), 0], [0, 0, 0, np.int64(22), np.int64(24), 0, 0], [0, 0, 0, 0, 0, np.int64(333), 0], [0, np.int64(23), np.int64(35), np.int64(53), np.int64(63), np.int64(38), 0], [0, np.int64(108), np.int64(194), np.int64(301), np.int64(387), np.int64(238), 0], [0, 0, 0, 0, 0, np.int64(1315), 0], [0, 0, 0, 0, np.int64(168), np.int64(992), 0], [0, 0, 0, 0, np.int64(30), np.int64(1090), 0], [0, np.int64(23), np.int64(33), np.int64(55), np.int64(50), np.int64(82), 0], [0, 0, np.int64(118), np.int64(164), np.int64(130), np.int64(575), 0], [0, 0, np.int64(22), np.int64(29), np.int64(59), np.int64(39), 0], [0, 0, np.int64(29), np.int64(67), np.int64(60), np.int64(29), 0], [0, np.int64(32), np.int64(34), np.int64(96), np.int64(136), 0, 0], [0, 0, 0, 0, 0, np.int64(427), 0], [0, np.int64(26), np.int64(88), np.int64(71), np.int64(63), np.int64(81), 0], [0, 0, 0, 0, 0, np.int64(288), 0], [0, np.int64(45), np.int64(45), np.int64(65), np.int64(56), np.int64(101), 0], [0, 0, np.int64(22), np.int64(25), 0, 0, 0], [0, np.int64(21), np.int64(21), np.int64(92), np.int64(80), np.int64(21), 0], [0, 0, 0, np.int64(30), np.int64(21), np.int64(132), 0], [0, np.int64(215), np.int64(314), np.int64(528), np.int64(853), np.int64(221), 0], [0, np.int64(66), 0, 0, np.int64(49), np.int64(477), 0], [0, np.int64(123), np.int64(141), np.int64(405), np.int64(250), np.int64(179), 0], [0, np.int64(178), 0, 0, np.int64(819), np.int64(396), 0], [0, np.int64(226), np.int64(254), np.int64(588), np.int64(408), np.int64(225), 0], [0, 0, np.int64(506), np.int64(1011), np.int64(419), np.int64(2628), 0], [0, 0, 0, 0, 0, np.int64(81), 0], [0, 0, np.int64(47), np.int64(55), np.int64(28), 0, 0], [0, np.int64(27), np.int64(21), 0, np.int64(224), np.int64(161), 0], [0, 0, 0, np.int64(70), 0, np.int64(156), 0], [0, np.int64(130), np.int64(143), np.int64(134), np.int64(201), np.int64(319), 0], [0, np.int64(25), 0, np.int64(104), np.int64(55), np.int64(38), 0], [np.int64(22), np.int64(506), np.int64(668), np.int64(2063), np.int64(2303), np.int64(883), 0], [0, 0, np.int64(28), np.int64(92), np.int64(59), np.int64(44), 0], [0, 0, np.int64(264), np.int64(325), np.int64(450), np.int64(373), 0], [0, np.int64(22), np.int64(56), np.int64(72), np.int64(85), np.int64(125), 0], [0, np.int64(28), np.int64(21), np.int64(94), np.int64(136), np.int64(57), 0], [np.int64(52), np.int64(224), np.int64(361), np.int64(444), np.int64(1499), np.int64(586), 0], [0, np.int64(30), np.int64(67), np.int64(105), np.int64(85), np.int64(40), 0], [0, 0, 0, 0, np.int64(27), np.int64(21), 0], [0, np.int64(484), np.int64(321), np.int64(1120), np.int64(1120), np.int64(191), 0], [0, np.int64(202), np.int64(152), np.int64(858), np.int64(514), np.int64(414), 0], [0, np.int64(32), np.int64(94), np.int64(49), np.int64(179), np.int64(43), 0], [0, np.int64(441), np.int64(589), np.int64(1020), np.int64(1608), np.int64(345), 0], [0, np.int64(51), np.int64(218), np.int64(163), np.int64(321), np.int64(138), 0], [0, 0, np.int64(21), np.int64(60), np.int64(98), 0, 0], [0, np.int64(27), np.int64(34), np.int64(89), 0, np.int64(28), 0], [np.int64(33), np.int64(401), np.int64(586), np.int64(1234), np.int64(1256), np.int64(196), 0], [0, np.int64(44), np.int64(192), np.int64(393), np.int64(840), np.int64(623), 0], [0, np.int64(315), np.int64(70), np.int64(849), np.int64(347), np.int64(536), 0], [0, np.int64(267), np.int64(574), np.int64(984), np.int64(1502), np.int64(1504), 0], [0, np.int64(93), np.int64(204), np.int64(187), np.int64(622), np.int64(200), 0], [0, np.int64(117), 0, np.int64(477), np.int64(475), np.int64(572), 0], [0, 0, 0, 0, 0, np.int64(310), 0], [0, np.int64(23), np.int64(102), 0, np.int64(328), np.int64(199), 0], [0, np.int64(137), np.int64(260), np.int64(842), np.int64(1528), np.int64(1052), 0], [0, np.int64(165), np.int64(336), np.int64(441), np.int64(911), np.int64(947), 0], [0, 0, 0, 0, 0, np.int64(312), 0], [0, 0, 0, 0, 0, np.int64(497), 0], [0, np.int64(69), 0, np.int64(373), np.int64(150), np.int64(102), 0], [0, np.int64(210), np.int64(188), np.int64(27), np.int64(672), np.int64(280), 0], [0, 0, 0, 0, np.int64(414), np.int64(301), 0], [0, np.int64(103), 0, np.int64(633), np.int64(129), np.int64(569), 0], [0, np.int64(22), 0, np.int64(90), np.int64(50), np.int64(40), 0], [0, 0, 0, 0, np.int64(123), np.int64(69), 0], [0, 0, 0, 0, np.int64(254), np.int64(182), 0], [0, 0, 0, 0, np.int64(182), np.int64(94), 0], [0, np.int64(24), np.int64(23), np.int64(52), np.int64(29), np.int64(25), 0], [0, 0, 0, 0, 0, np.int64(250), 0], [0, 0, np.int64(45), np.int64(40), np.int64(49), np.int64(132), 0], [0, 0, 0, 0, 0, np.int64(178), 0], [0, 0, 0, 0, 0, np.int64(646), 0], [0, 0, np.int64(105), np.int64(64), np.int64(74), np.int64(79), 0], [0, np.int64(21), np.int64(34), np.int64(34), np.int64(74), np.int64(27), 0], [0, 0, 0, np.int64(175), 0, np.int64(113), 0], [0, 0, 0, np.int64(38), 0, np.int64(41), 0], [0, 0, 0, 0, 0, np.int64(314), 0], [0, 0, 0, np.int64(38), 0, np.int64(91), 0], [0, 0, np.int64(33), np.int64(45), np.int64(35), np.int64(55), 0], [0, 0, np.int64(40), 0, np.int64(51), np.int64(33), 0], [0, 0, 0, np.int64(33), np.int64(29), np.int64(21), 0], [0, 0, 0, 0, 0, np.int64(377), 0], [0, np.int64(22), np.int64(27), np.int64(32), np.int64(101), 0, 0]]

demand_all_from_input_matrix = {}
for i, hub_id_str in enumerate(hub_ids_str_list):
    demand_all_from_input_matrix[hub_id_str] = demand_matrix_raw[i]

# Create a consolidated demand dictionary mapping numeric hub ID to its list of hourly demands
consolidated_demands_by_hub_and_hour = {
    int(key.split('_')[1]): [int(d) for d in value_list] # Convert np.int64 to int
    for key, value_list in demand_all_from_input_matrix.items()
}

print(f"Successfully processed provided demand data. Found {len(consolidated_demands_by_hub_and_hour)} specific hubs with demands.")

# Define specific demands for each hour based on the provided matrix columns
hourly_specific_demands = {}
# For hour H (0-6), use demands from the (H)th column (index H) of the matrix.
for h_idx in range(7): # Covers hour 0 to hour 6 (as per matrix columns 0-6)
    hourly_specific_demands[h_idx] = {
        h: consolidated_demands_by_hub_and_hour.get(h, [0]*7)[h_idx] # Access demand at actual hour index
        if h_idx < len(consolidated_demands_by_hub_and_hour.get(h, [0]*7)) else 0 # Defensive indexing
        for h in HUBS
    }

previous_active_vehicles_info = {}
all_vehicles_ever_used_ids = set() # Global set to track all vehicle IDs

total_overall_cost = 0.0
hourly_results_summary = []

# Data for plotting
vehicles_used_per_hour = {}
max_time_per_vehicle_per_hour = {}
total_hubs_visited_per_hour = {} # Changed to count all visits

hour_label_to_index = {
    0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5,
    "6:00-7:00 AM": 6, # This is now hour_idx 6 in the overall sequence
    "7:00-8:00 AM": 7
}

# Mapping hour_idx to time limits for clarity
time_limits_per_hour_idx = {
    0: MAX_DELIVERY_TIME_PER_VEHICLE_0_AM_TO_1_AM,
    1: MAX_DELIVERY_TIME_PER_VEHICLE_1_AM_TO_2_AM,
    2: MAX_DELIVERY_TIME_PER_VEHICLE_2_AM_TO_3_AM,
    3: MAX_DELIVERY_TIME_PER_VEHICLE_3_AM_TO_4_AM,
    4: MAX_DELIVERY_TIME_PER_VEHICLE_4_AM_TO_5_AM,
    5: MAX_DELIVERY_TIME_PER_VEHICLE_5_AM_TO_6_AM,
    6: MAX_DELIVERY_TIME_PER_VEHICLE_6_AM_TO_7_AM, # Changed key to integer 6
    7: MAX_DELIVERY_TIME_PER_VEHICLE_7_AM_TO_8_AM # Changed key to integer 7
}


# Main simulation loop for cycles 0 AM to 5 AM (hour_idx 0 to 5)
for hour_idx in range(6):
    current_hour_label_for_display = f"{hour_idx}:00-{hour_idx+1}:00 AM" # For plot labels
    current_hour_numeric_idx = hour_idx

    current_demands_for_hour = hourly_specific_demands[hour_idx].copy() # Get demands from the pre-defined dict

    print(f"\n--- Starting ALNS Optimization for Hour {current_hour_label_for_display} ---")

    # Get time limit for the current hour from the new mapping
    time_limit_for_this_hour = time_limits_per_hour_idx.get(current_hour_numeric_idx)

    time_matrix_to_use = original_time_matrix

    result = solve_vrp_with_alns({
                                    'hour': current_hour_label_for_display, # Pass formatted label for plotting
                                    'demands': current_demands_for_hour,
                                    'hour_index': current_hour_numeric_idx
                                },
                                    distances,
                                    time_matrix_to_use,
                                    time_limit_for_this_hour, # Apply the time limit for this hour
                                    previous_active_vehicles_info,
                                    is_final_push=False)

    hourly_results_summary.append(result)

    if result['status'] == 'ALNS_COMPLETED' or result['status'] == 'NO_DEMAND':
        total_overall_cost += result['objective_value']
        previous_active_vehicles_info = result['active_vehicles_for_next_hour']

        print_summary_and_plot_results(current_hour_label_for_display, result, node_coordinates) # Pass formatted label
        unfulfilled_demand_sum = sum(result['unfulfilled_demand_per_hub'].values())
        print(f"  Unfulfilled Demand from {current_hour_label_for_display} cycle (sum): {unfulfilled_demand_sum:.2f}")

        # Collect data for plotting
        vehicles_used_per_hour[current_hour_label_for_display] = result['used_vehicles_count'] # Use formatted label
        max_time = 0
        if result['routes']:
            # Filter routes by total_delivered safely using .get()
            active_routes_in_result = [r for r in result['routes'] if r.get('total_delivered', 0) > 0]
            if active_routes_in_result: # Only take max if there are active routes
                max_time = max(r['time'] for r in active_routes_in_result)
        max_time_per_vehicle_per_hour[current_hour_label_for_display] = max_time # Use formatted label

        # Count total hubs visited (including duplicates)
        total_hubs_visited_this_hour_count = 0
        for delivery_info in result['deliveries']:
            for hub_id, qty in delivery_info['deliveries'].items():
                if qty > 0:
                    total_hubs_visited_this_hour_count += 1 # Count each visit
        total_hubs_visited_per_hour[current_hour_label_for_display] = total_hubs_visited_this_hour_count # Use formatted label


    else:
        previous_active_vehicles_info = {}
        print(f"ALNS optimization failed or had issues for Hour {current_hour_label_for_display}. Status: {result['status']}")


# --- Special Handling for 6:00 AM to 7:00 AM Cycle ---
current_hour_6_00_to_7_00_label = "6:00-7:00 AM"
current_hour_6_00_to_7_00_idx = hour_label_to_index[current_hour_6_00_to_7_00_label]

# Demand for 6:00-7:00 AM cycle now comes from the consolidated_demands_by_hub_and_hour (index 6)
current_demands_for_6_00_to_7_00 = hourly_specific_demands[6].copy()


print(f"\n--- Starting ALNS Optimization for {current_hour_6_00_to_7_00_label} ---")

# Get time limit for the 6:00-7:00 AM cycle
time_limit_for_this_cycle_6_7 = time_limits_per_hour_idx.get(current_hour_6_00_to_7_00_idx)
time_matrix_to_use_6_7 = original_time_matrix

result_6_00_to_7_00 = solve_vrp_with_alns({
                                            'hour': current_hour_6_00_to_7_00_label,
                                            'demands': current_demands_for_6_00_to_7_00,
                                            'hour_index': current_hour_6_00_to_7_00_idx
                                        },
                                            distances,
                                            time_matrix_to_use_6_7,
                                            time_limit_for_this_cycle_6_7,
                                            previous_active_vehicles_info,
                                            is_final_push=False)

hourly_results_summary.append(result_6_00_to_7_00)

unfulfilled_demand_from_6_00_to_7_00 = {j: 0.0 for j in HUBS}
if result_6_00_to_7_00['status'] == 'ALNS_COMPLETED' or result_6_00_to_7_00['status'] == 'NO_DEMAND':
    total_overall_cost += result_6_00_to_7_00['objective_value']
    previous_active_vehicles_info = result_6_00_to_7_00['active_vehicles_for_next_hour']
    unfulfilled_demand_from_6_00_to_7_00 = result_6_00_to_7_00['unfulfilled_demand_per_hub']

    print_summary_and_plot_results(current_hour_6_00_to_7_00_label, result_6_00_to_7_00, node_coordinates)
    unfulfilled_demand_sum_6_7 = sum(unfulfilled_demand_from_6_00_to_7_00.values())
    print(f"  Unfulfilled Demand from {current_hour_6_00_to_7_00_label} cycle (sum): {unfulfilled_demand_sum_6_7:.2f}")

    # Collect data for plotting
    vehicles_used_per_hour[current_hour_6_00_to_7_00_label] = result_6_00_to_7_00['used_vehicles_count']
    max_time = 0
    if result_6_00_to_7_00['routes']:
        active_routes_in_result_6_7 = [r for r in result_6_00_to_7_00['routes'] if r.get('total_delivered', 0) > 0]
        if active_routes_in_result_6_7:
            max_time = max(r['time'] for r in active_routes_in_result_6_7)
    max_time_per_vehicle_per_hour[current_hour_6_00_to_7_00_label] = max_time

    # Count total hubs visited (including duplicates)
    total_hubs_visited_this_hour_count = 0
    for delivery_info in result_6_00_to_7_00['deliveries']:
        for hub_id, qty in delivery_info['deliveries'].items():
            if qty > 0:
                total_hubs_visited_this_hour_count += 1
    total_hubs_visited_per_hour[current_hour_6_00_to_7_00_label] = total_hubs_visited_this_hour_count

else:
    previous_active_vehicles_info = {}
    print(f"ALNS optimization failed or had issues for {current_hour_6_00_to_7_00_label}. Status: {result_6_00_to_7_00['status']}")
    # If optimization failed, assume all demand from this cycle is unfulfilled for the next cycle
    unfulfilled_demand_from_6_00_to_7_00 = {j: current_demands_for_6_00_to_7_00[j] for j in HUBS if j in current_demands_for_6_00_to_7_00}


# --- Special Handling for 7:00 AM to 8:00 AM Cycle (Final Push) ---
# Removed fixed_demand_7_00_to_8_00_specific_hubs as per user's request.
# Demand for Hour 7 (index 7) from the matrix is 0 as it doesn't exist.
current_demands_for_7_00_to_8_00 = defaultdict(float)
# Add unfulfilled demand from previous cycle
for hub_id, qty in unfulfilled_demand_from_6_00_to_7_00.items():
    current_demands_for_7_00_to_8_00[hub_id] += qty

# Filter out hubs with 0 demand if any were initialized but not needed
current_demands_for_7_00_to_8_00 = {h: d for h, d in current_demands_for_7_00_to_8_00.items() if d > 0.001}


if sum(current_demands_for_7_00_to_8_00.values()) > 0.001:
    current_hour_7_00_to_8_00_label = "7:00-8:00 AM"
    current_hour_7_00_to_8_00_idx = hour_label_to_index[current_hour_7_00_to_8_00_label]

    print(f"\n--- Starting FINAL ALNS Optimization for {current_hour_7_00_to_8_00_label} (Remaining & New Demands) ---")

    time_matrix_to_use_7_8 = time_zero_matrix
    # Get time limit for the 7:00-8:00 AM cycle
    time_limit_for_this_cycle_7_8 = time_limits_per_hour_idx.get(current_hour_7_00_to_8_00_idx)

    result_7_00_to_8_00 = solve_vrp_with_alns({
                                                'hour': current_hour_7_00_to_8_00_label,
                                                'demands': current_demands_for_7_00_to_8_00,
                                                'hour_index': current_hour_7_00_to_8_00_idx
                                            },
                                                distances,
                                                time_matrix_to_use_7_8,
                                                time_limit_for_this_cycle_7_8,
                                                previous_active_vehicles_info,
                                                is_final_push=True)

    hourly_results_summary.append(result_7_00_to_8_00)

    if result_7_00_to_8_00['status'] == 'ALNS_COMPLETED' or result_7_00_to_8_00['status'] == 'NO_DEMAND':
        total_overall_cost += result_7_00_to_8_00['objective_value']

        print_summary_and_plot_results(current_hour_7_00_to_8_00_label, result_7_00_to_8_00, node_coordinates)
        unfulfilled_demand_sum_7_8 = sum(result_7_00_to_8_00['unfulfilled_demand_per_hub'].values())
        print(f"  Unfulfilled Demand from {current_hour_7_00_to_8_00_label} cycle (sum): {unfulfilled_demand_sum_7_8:.2f}")

        # Collect data for plotting
        vehicles_used_per_hour[current_hour_7_00_to_8_00_label] = result_7_00_to_8_00['used_vehicles_count']
        max_time = 0
        if result_7_00_to_8_00['routes']:
            active_routes_in_result_7_8 = [r for r in result_7_00_to_8_00['routes'] if r.get('total_delivered', 0) > 0]
            if active_routes_in_result_7_8:
                max_time = max(r['time'] for r in active_routes_in_result_7_8)
        max_time_per_vehicle_per_hour[current_hour_7_00_to_8_00_label] = max_time

        # Count total hubs visited (including duplicates)
        total_hubs_visited_this_hour_count = 0
        for delivery_info in result_7_00_to_8_00['deliveries']:
            for hub_id, qty in delivery_info['deliveries'].items():
                if qty > 0:
                    total_hubs_visited_this_hour_count += 1
        total_hubs_visited_per_hour[current_hour_7_00_to_8_00_label] = total_hubs_visited_this_hour_count

    else:
        print(f"ALNS optimization failed or had issues for {current_hour_7_00_to_8_00_label}. Status: {result_7_00_to_8_00['status']}")
else:
    print("\n--- No significant unfulfilled demand or new demand for 7:00-8:00 AM cycle. Skipping final push. ---")


# --- Final Summary ---
print("\n" + "="*50)
print(f"Total Overall Cost for all operations: {total_overall_cost:.2f} yuan")
print(f"Number of distinct vehicles used throughout: {len(all_vehicles_ever_used_ids)}")
print("="*50)

# Optional: Print detailed summary of each hour
print("\n--- Detailed Hourly Results Summary ---")
for res in hourly_results_summary:
    hour_display = res.get('hour', 'N/A')
    total_unfulfilled = sum(res['unfulfilled_demand_per_hub'].values())

    # Calculate total hubs visited for the final summary print (this is already collected in total_hubs_visited_per_hour)
    # Re-calculating here for consistency with the loop if needed
    hubs_visited_for_summary_count = 0
    for delivery_info in res['deliveries']:
        for hub_id, qty in delivery_info['deliveries'].items():
            if qty > 0:
                hubs_visited_for_summary_count += 1

    if res['status'] == 'ALNS_COMPLETED' or res['status'] == 'NO_DEMAND':
        print(f"Hour {hour_display}: Cost = {res['objective_value']:.2f} yuan, Used Vehicles = {res['used_vehicles_count']}, Unfulfilled Demand = {total_unfulfilled:.2f}, Total Hubs Visited (incl. duplicates) = {hubs_visited_for_summary_count}")
    else:
        print(f"Hour {hour_display}: Optimization Status = {res['status']}, No Solution Found")

# --- Vehicle Reuse Plot ---
print("\n--- Vehicle Reuse Analysis ---")

# Filter for vehicles used more than once
reused_vehicles = {v_id: count for v_id, count in vehicle_usage_counts.items() if count > 1}

if reused_vehicles:
    # Sort for better visualization
    sorted_reused_vehicles = sorted(reused_vehicles.items(), key=lambda item: item[1], reverse=True)

    vehicle_ids = [f'Vehicle {v[0]+1}' for v in sorted_reused_vehicles]
    reuse_counts = [v[1] for v in sorted_reused_vehicles]

    plt.figure(figsize=(12, 7))
    plt.bar(vehicle_ids, reuse_counts, color='skyblue')
    plt.xlabel("Vehicle ID")
    plt.ylabel("Number of Cycles Used")
    plt.title("Vehicles Reused Across Multiple Cycles")
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show() # Changed from plt.savefig()
    # plt.close() # Removed plt.close() to keep plots open for display

    print("\nVehicles reused in more than one cycle:")
    for v_id, count in sorted_reused_vehicles:
        print(f"  Vehicle {v_id+1}: Used in {count} cycles")
else:
    print("No vehicles were reused in more than one cycle.")

# --- New Visualizations ---
print("\n--- Generating Performance Charts ---")

#### 1. Pie Chart: Number of Vehicles Used Every Hour
def plot_vehicles_used_pie(vehicles_used_data):
    """Plots a pie chart of the number of vehicles used per hour."""
    if not vehicles_used_data:
        print("No vehicle usage data to plot.")
        return

    labels = list(vehicles_used_data.keys())
    sizes = list(vehicles_used_data.values())

    plt.figure(figsize=(12, 12))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=plt.cm.Paired.colors, textprops={'fontsize': 10})
    plt.axis('equal')
    plt.title("Proportion of Vehicles Used Per Hour", fontsize=14)
    plt.tight_layout()
    plt.show() # Changed from plt.savefig()
    # plt.close() # Removed plt.close() to keep plots open for display

plot_vehicles_used_pie(vehicles_used_per_hour)

#### 2. Bar Chart: Max Time per Vehicle per Hour vs. Time Limit
def plot_max_time_per_vehicle_vs_limit(max_time_data, time_limits_data, hour_label_to_idx_map):
    """Plots a bar chart comparing max vehicle time per hour with the set time limit."""
    if not max_time_data:
        print("No max time per vehicle data to plot.")
        return

    hours = list(max_time_data.keys())
    max_times = list(max_time_data.values())

    # Map the display labels back to their numeric index for time limit lookup
    corresponding_time_limits = [time_limits_data.get(hour_label_to_idx_map.get(h, None) if isinstance(h, str) else h, 0) for h in hours]


    x = np.arange(len(hours))
    width = 0.35

    fig, ax = plt.subplots(figsize=(14, 8))
    rects1 = ax.bar(x - width/2, max_times, width, label='Max Vehicle Time (minutes)', color='skyblue')
    rects2 = ax.bar(x + width/2, corresponding_time_limits, width, label='Time Limit (minutes)', color='lightcoral', alpha=0.7)

    ax.set_xlabel("Hour Cycle")
    ax.set_ylabel("Time (minutes)")
    ax.set_title("Maximum Vehicle Time per Hour vs. Time Limit")
    ax.set_xticks(x)
    ax.set_xticklabels(hours, rotation=45, ha='right') # Use the hour display labels
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show() # Changed from plt.savefig()
    # plt.close() # Removed plt.close() to keep plots open for display

plot_max_time_per_vehicle_vs_limit(max_time_per_vehicle_per_hour, time_limits_per_hour_idx, hour_label_to_index)

#### 3. Bar Chart: Total Hubs Visited per Hour (including duplicates)
def plot_total_hubs_visited_per_hour(total_hubs_visited_data):
    """Plots a bar chart of the total number of hubs visited per hour (including duplicates)."""
    if not total_hubs_visited_data:
        print("No total hubs visited data to plot.")
        return

    labels = list(total_hubs_visited_data.keys())
    num_hubs = list(total_hubs_visited_data.values())

    plt.figure(figsize=(12, 7))
    plt.bar(labels, num_hubs, color='lightgreen')
    plt.xlabel("Hour Cycle")
    plt.ylabel("Total Number of Hubs Visited (incl. duplicates)")
    plt.title("Total Number of Hubs Visited Per Hour Cycle")
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show() # Changed from plt.savefig()
    # plt.close() # Removed plt.close() to keep plots open for display

plot_total_hubs_visited_per_hour(total_hubs_visited_per_hour)


#### 4. Bar Chart: Unfulfilled Demand per Hour
def plot_unfulfilled_demand_per_hour(hourly_results):
    """Plots a bar chart of total unfulfilled demand per hour."""
    hours = []
    unfulfilled_demands = []

    for result in hourly_results:
        hour_label = result.get('hour', 'N/A')
        total_unfulfilled = sum(result['unfulfilled_demand_per_hub'].values())
        hours.append(hour_label)
        unfulfilled_demands.append(total_unfulfilled)

    if not hours:
        print("No unfulfilled demand data to plot.")
        return

    plt.figure(figsize=(12, 7))
    plt.bar(hours, unfulfilled_demands, color='orange')
    plt.xlabel("Hour Cycle")
    plt.ylabel("Total Unfulfilled Demand (units)")
    plt.title("Total Unfulfilled Demand Per Hour Cycle")
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show() # Changed from plt.savefig()
    # plt.close() # Removed plt.close() to keep plots open for display

plot_unfulfilled_demand_per_hour(hourly_results_summary)

