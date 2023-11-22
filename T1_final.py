import json
import csv
from collections import defaultdict
import heapq
import time
import math
from datetime import datetime
from datetime import timedelta

# Graph Class
class Graph:
    def __init__(self):
        self.edges = defaultdict(dict)
        self.node_coordinates = {}

    def add_edge(self, from_node, to_node, length, speed_data):
        # speed_data is a dictionary containing speed for each hour
        self.edges[from_node][to_node] = {'length': length, 'speed': speed_data}
        
    def add_node_coordinates(self, node, coordinates):
        self.node_coordinates[node] = coordinates

# Load Edge Data
def load_edge_data(filename): #this pre-process the edge data connection to form the graph
    graph = Graph()
    with open(filename, newline='') as file:
        reader = csv.DictReader(file)
        for row in reader:
            start_node = int(row['start_id'])
            end_node = int(row['end_id'])
            length = float(row['length'])

            # Extract speed data for each hour
            speed_data = {k: float(row[k]) for k in reader.fieldnames if k.startswith('weekday_') or k.startswith('weekend_')}

            graph.add_edge(start_node, end_node, length, speed_data)
    return graph

def load_node_data(filename,graph):# this add the node coordinates to the graph
    with open(filename, 'r') as f:
        node_data = json.load(f)
        for node, coordinates in node_data.items():
            graph.add_node_coordinates(int(node), coordinates)
            
            
def find_nearest_2_nodes(graph, source_lat, source_long, dest_lat, dest_long):
    def euclidean_distance(coord1, coord2):
        lat1, lon1 = coord1['lat'], coord1['lon']
        lat2, lon2 = coord2['lat'], coord2['lon']
        return math.sqrt((lat1 - lat2) ** 2 + (lon1 - lon2) ** 2)

    min_dist_source = float('inf')
    nearest_source_node = None
    min_dist_destination = float('inf')
    nearest_destination_node = None

    for node, coordinates in graph.node_coordinates.items():
        dist_source = euclidean_distance({'lat': source_lat, 'lon': source_long}, coordinates)
        dist_destination = euclidean_distance({'lat': dest_lat, 'lon': dest_long}, coordinates)
   
        if dist_source < min_dist_source:
            min_dist_source = dist_source
            nearest_source_node = node
        if dist_destination < min_dist_destination:
            min_dist_destination = dist_destination
            nearest_destination_node = node

    return nearest_source_node, nearest_destination_node

def find_nearest_nodes(graph, source_lat, source_long):
    def euclidean_distance(coord1, coord2):
        lat1, lon1 = coord1['lat'], coord1['lon']
        lat2, lon2 = coord2['lat'], coord2['lon']
        return math.sqrt((lat1 - lat2) ** 2 + (lon1 - lon2) ** 2)

    min_dist = float('inf')
    nearest_node = None

    for node, coordinates in graph.node_coordinates.items():
        dist = euclidean_distance({'lat': source_lat, 'lon': source_long}, coordinates)
        if dist < min_dist:
            min_dist = dist
            nearest_node = node

    return nearest_node



def date_to_time_of_day(date_time_str):
    if isinstance(date_time_str, str):
    # Convert the string to a datetime object
        date_time_obj = datetime.strptime(date_time_str, "%m/%d/%Y %H:%M:%S")
    else:
        date_time_obj = date_time_str
    hour = int(date_time_obj.strftime("%H"))  # Get hour in 24-hour format
    day_of_week = date_time_obj.strftime("%A").lower()  # Get day of the week

    # Determine if it's a weekday or weekend
    if day_of_week in ['saturday', 'sunday']:
        return f"weekend_{hour}"
    else:
        return f"weekday_{hour}"
    
    

def dijkstra_time(graph, start_node, end_node, time_of_day):
    def get_travel_time(from_node, to_node):
        edge_data = graph.edges[from_node][to_node]
        distance = edge_data['length']
        
        #print(f"Available speed keys: {list(edge_data['speed'].keys())}, Queried key: {time_of_day}")
        
        
        speed = edge_data['speed'][time_of_day] if time_of_day in edge_data['speed'] else edge_data['speed']['default']
        return distance / max(speed, 1)  # Avoid division by zero

    distances = {node: float('infinity') for node in graph.edges}
    previous_nodes = {node: None for node in graph.edges}
    distances[start_node] = 0
    queue = [(0, start_node)]

    while queue:
        current_distance, current_node = heapq.heappop(queue)
        if current_node == end_node:
            break

        for neighbor in graph.edges[current_node]:
            travel_time = get_travel_time(current_node, neighbor)
            distance = current_distance + travel_time
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                previous_nodes[neighbor] = current_node
                heapq.heappush(queue, (distance, neighbor))

    # Reconstruct the shortest path and calculate total travel time
    path, total_travel_time = [], 0
    current_node = end_node
    while current_node:
        path.append(current_node)
        next_node = previous_nodes[current_node]
        if next_node:
            total_travel_time += get_travel_time(next_node, current_node)
        current_node = next_node
    path.reverse()

    return path, total_travel_time

def calculate_total_travel_time_location(graph, date_time, driver_lat, driver_lon, pickup_lat, pickup_lon, dropoff_lat, dropoff_lon):
    time_of_day = date_to_time_of_day(date_time)

    # Find the nearest nodes for the driver, pickup, and dropoff locations
    driver_node = find_nearest_nodes(graph, driver_lat, driver_lon)
    pickup_node = find_nearest_nodes(graph, pickup_lat, pickup_lon)
    dropoff_node = find_nearest_nodes(graph, dropoff_lat, dropoff_lon)

    # Calculate the route and time from the driver to the pickup location
    _, time_to_pickup = dijkstra_time(graph, driver_node, pickup_node, time_of_day)

    # Calculate the route and time from the pickup location to the dropoff location
    _, time_to_dropoff = dijkstra_time(graph, pickup_node, dropoff_node, time_of_day)

    return time_to_pickup + time_to_dropoff


def calculate_total_travel_time_node(graph, date_time, driver_node, pickup_node, dropoff_node):
    time_of_day = date_to_time_of_day(date_time)

    # Calculate the route and time from the driver to the pickup location
    _, time_to_pickup = dijkstra_time(graph, driver_node, pickup_node, time_of_day)

    # Calculate the route and time from the pickup location to the dropoff location
    _, time_to_dropoff = dijkstra_time(graph, pickup_node, dropoff_node, time_of_day)
    return time_to_pickup + time_to_dropoff, time_to_pickup, time_to_dropoff







# Simple Baseline Algorithm Class
class SimpleBaselineAlgorithm:
    def __init__(self, graph):
        self.graph = graph
        self.available_drivers = []
        self.unmatched_passengers = []
        self.total_matches = 0
        self.total_travel_time = 0
        self.pickup=0
        self.dropoff=0

    def add_driver(self, time, node):
        heapq.heappush(self.available_drivers, (time, node))

    def add_passenger(self, time, pickup_node, dropoff_node):
        heapq.heappush(self.unmatched_passengers, (time, pickup_node, dropoff_node))

    def match_passenger_to_driver(self):
        if self.available_drivers and self.unmatched_passengers:
            available_time, driver_location = heapq.heappop(self.available_drivers)
            request_time, pickup_location, dropoff_location = heapq.heappop(self.unmatched_passengers)

            # Calculate travel times using existing functions
            #time_to_pickup = calculate_total_travel_time_node(self.graph, available_time, driver_location, pickup_location)
            #time_to_dropoff = calculate_total_travel_time_node(self.graph, available_time, pickup_location, dropoff_location)

            total_travel_time, time_to_pickup, time_to_dropoff = calculate_total_travel_time_node(self.graph, available_time, driver_location, pickup_location,dropoff_location)
            self.total_matches += 1
            self.total_travel_time += total_travel_time
            self.pickup+=time_to_pickup
            self.dropoff+=time_to_dropoff

            # Update the driver's availability time
            new_available_time = request_time + timedelta(seconds=total_travel_time)
            heapq.heappush(self.available_drivers, (new_available_time, dropoff_location))
            return total_travel_time

        return None

    
    def report_stats(self):
        print(f"Total matches made: {self.total_matches}")
        print(f"Total travel time for all matches: {self.total_travel_time}")
        print(f"Total time to pick up:{self.pickup}")
        print(f"Total time to drop off:{self.dropoff} ")
        print(f"Passengers' total travel time:{self.pickup+self.dropoff}")
        print(f"Drivers' profit:{self.dropoff-self.pickup}")

# Load Drivers and Passengers from CSV
def load_drivers(filename, graph):
    drivers = []
    with open(filename, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            date_time = datetime.strptime(row['Date/Time'], "%m/%d/%Y %H:%M:%S")
            source_lat = float(row['Source Lat'])
            source_lon = float(row['Source Lon'])
            nearest_node = find_nearest_nodes(graph, source_lat, source_lon)
            drivers.append((date_time, nearest_node))
    return drivers



def load_passengers(filename,graph):
    passengers = []
    with open(filename, newline='') as file:
        reader = csv.DictReader(file)
        for row in reader:
            date_time = datetime.strptime(row['Date/Time'], "%m/%d/%Y %H:%M:%S")
            source_lat = float(row['Source Lat'])
            source_lon = float(row['Source Lon'])
            dest_lat = float(row['Dest Lat'])
            dest_lon = float(row['Dest Lon'])
            source_node, dest_node = find_nearest_2_nodes(graph, source_lat, source_lon, dest_lat, dest_lon)
            passengers.append((date_time, source_node, dest_node))
    return passengers


def main():
    graph = load_edge_data("edges.csv")
    
    load_node_data("node_data.json",graph)
    
    for start_node in list(graph.edges.keys())[:1]:
        for end_node in graph.edges[start_node]:
            edge_data = graph.edges[start_node][end_node]
            print(f"Edge from {start_node} to {end_node}: Length = {edge_data['length']}, Speed Data = {edge_data['speed']}")
            
    for node_id in list(graph.node_coordinates.keys())[:1]:
        print(f"Node {node_id} has coordinates {graph.node_coordinates[node_id]}")
        
        
    nearest_source_node, nearest_dest_node = find_nearest_2_nodes(graph, 40.6466, -73.7896, 40.7603, -73.9794)
    print(f"Nearest source node: {nearest_source_node}, Nearest destination node: {nearest_dest_node}")
    
    time_test=calculate_total_travel_time_location(graph, '04/25/2014 00:14:00', 40.667, -73.8713, 40.6466, -73.7896, 40.7603, -73.9794)    
    print(f"Total travel time: {time_test}")
    
    
    
def T1():
    start_time = time.time()
    graph = load_edge_data("edges.csv")
    load_node_data("node_data.json",graph)
    algorithm = SimpleBaselineAlgorithm(graph)
    
    drivers = load_drivers("drivers.csv", graph)
    for driver_time, driver_node in drivers:
        algorithm.add_driver(driver_time, driver_node)

    passengers = load_passengers("passengers.csv", graph)
    for passenger in passengers:
        algorithm.add_passenger(*passenger)

    while algorithm.unmatched_passengers:
        travel_time = algorithm.match_passenger_to_driver()
        

    algorithm.report_stats()

    
    end_time = time.time()
    print(f"Program running time: {end_time - start_time} seconds")

T1()