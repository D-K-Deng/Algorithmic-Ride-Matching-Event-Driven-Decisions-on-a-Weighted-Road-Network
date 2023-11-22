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


def euclidean_distance(coord1, coord2):
    lat1, lon1 = coord1['lat'], coord1['lon']
    lat2, lon2 = coord2['lat'], coord2['lon']
    return math.sqrt((lat1 - lat2) ** 2 + (lon1 - lon2) ** 2)

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
    
def format_time_of_day(date_time):
    hour = date_time.hour  # Extract hour from datetime
    day_of_week = date_time.strftime("%A").lower()  # Get day of the week

    # Format time_of_day as per your speed data keys
    if day_of_week in ['saturday', 'sunday']:
        return f"weekend_{hour}"
    else:
        return f"weekday_{hour}"

   

def dijkstra_time_estimate(graph, start_node, end_node, time_of_day):
    def get_travel_time(from_node, to_node):
        edge_data = graph.edges[from_node][to_node]
        distance = edge_data['length']
        speed = edge_data['speed'][time_of_day] if time_of_day in edge_data['speed'] else edge_data['speed']['weekend_0']
        return distance / max(speed, 1)  # Avoid division by zero

    travel_times = {node: float('infinity') for node in graph.edges}
    travel_times[start_node] = 0
    priority_queue = [(0, start_node)]

    while priority_queue:
        current_travel_time, current_node = heapq.heappop(priority_queue)
        if current_node == end_node:
            return current_travel_time

        for neighbor in graph.edges[current_node]:
            edge_travel_time = get_travel_time(current_node, neighbor)
            new_travel_time = current_travel_time + edge_travel_time
            if new_travel_time < travel_times[neighbor]:
                travel_times[neighbor] = new_travel_time
                heapq.heappush(priority_queue, (new_travel_time, neighbor))
                     
    return float('inf')



def calculate_total_travel_time_location(graph, date_time, driver_lat, driver_lon, pickup_lat, pickup_lon, dropoff_lat, dropoff_lon):
    time_of_day = date_to_time_of_day(date_time)

    # Find the nearest nodes for the driver, pickup, and dropoff locations
    driver_node = find_nearest_nodes(graph, driver_lat, driver_lon)
    pickup_node = find_nearest_nodes(graph, pickup_lat, pickup_lon)
    dropoff_node = find_nearest_nodes(graph, dropoff_lat, dropoff_lon)

    # Calculate the route and time from the driver to the pickup location
    _, time_to_pickup = dijkstra_time_estimate(graph, driver_node, pickup_node, time_of_day)

    # Calculate the route and time from the pickup location to the dropoff location
    _, time_to_dropoff = dijkstra_time_estimate(graph, pickup_node, dropoff_node, time_of_day)

    return time_to_pickup + time_to_dropoff


def calculate_total_travel_time_node(graph, date_time, driver_node, pickup_node, dropoff_node):
    
    time_of_day = date_to_time_of_day(date_time)

    # Calculate the route and time from the driver to the pickup location
    time_to_pickup = dijkstra_time_estimate(graph, driver_node, pickup_node, time_of_day)

    # Calculate the route and time from the pickup location to the dropoff location
    time_to_dropoff = dijkstra_time_estimate(graph, pickup_node, dropoff_node, time_of_day)

    return time_to_pickup + time_to_dropoff

# Estimate time Algorithm Class
class EstimateTimeAlgorithm:
    def __init__(self, graph):
        self.graph = graph
        self.available_drivers = []
        self.unmatched_passengers = []
        self.driver_heap = []
        self.total_matches = 0
        self.total_travel_time = 0
        self.pickup = 0
        self.dropoff = 0

    def add_driver(self, time, node):
        heapq.heappush(self.available_drivers, (time, node))

    def add_passenger(self, time, pickup_node, dropoff_node):
        heapq.heappush(self.unmatched_passengers, (time, pickup_node, dropoff_node))
        
    def initialize_heap(self):
        # Called once to initialize the heap
        self.driver_heap = [(driver[0], driver[1]) for driver in self.available_drivers]
        heapq.heapify(self.driver_heap)

    def match_passenger_to_driver(self):

        if self.available_drivers and self.unmatched_passengers:
            passenger_time, pickup_location, dropoff_location = heapq.heappop(self.unmatched_passengers)
            
            top_drivers = heapq.nsmallest(10, self.available_drivers)
            
            
            min_time = float('inf')
            min_driver = None
            min_index = None
            for index, driver in enumerate(top_drivers):
                travel_time = calculate_total_travel_time_node(self.graph, passenger_time, driver[1], pickup_location, dropoff_location)
                if travel_time < min_time:
                    min_time = travel_time
                    min_driver = driver

            travel_time_to_pickup = dijkstra_time_estimate(self.graph, min_driver[1], pickup_location, min_driver[0])
            travel_time_to_dropoff = dijkstra_time_estimate(self.graph, pickup_location, dropoff_location, min_driver[0] + timedelta(seconds=travel_time_to_pickup))
            total_travel_time = travel_time_to_pickup + travel_time_to_dropoff
            new_available_time = min_driver[0] + timedelta(seconds=total_travel_time)

            heapq.heappush(self.driver_heap, (new_available_time, dropoff_location))


            # Update total matches and travel time

            self.total_matches += 1
            self.pickup += travel_time_to_pickup
            self.dropoff += travel_time_to_dropoff
            self.total_travel_time += total_travel_time

            return total_travel_time
        return None
    
    def report_stats(self):
        print(f"Total matches made: {self.total_matches}")
        print(f"Total travel time for all matches: {self.total_travel_time}")
        print(f"Total time to pick up: {self.pickup}")
        print(f"Total time to drop off: {self.dropoff}")
        print(f"Passengers' total travel time: {self.pickup + self.dropoff}")
        print(f"Drivers' profit:{self.dropoff - self.pickup}")

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


    
    
def T3():
    start_time = time.time()
    graph = load_edge_data("edges.csv")
    load_node_data("node_data.json",graph)
    algorithm = EstimateTimeAlgorithm(graph)
    
    
    drivers = load_drivers("drivers.csv", graph)
    for driver_time, driver_node in drivers:
        algorithm.add_driver(driver_time, driver_node)

    passengers = load_passengers("passengers.csv", graph)
    for passenger in passengers:
        algorithm.add_passenger(*passenger)
        
    #algorithm.initialize_heap()
    print("preprocessing done")

    while algorithm.unmatched_passengers:
        travel_time = algorithm.match_passenger_to_driver()


    algorithm.report_stats()

    
    end_time = time.time()
    print(f"Program running time: {end_time - start_time} seconds")    
    

    
T3()