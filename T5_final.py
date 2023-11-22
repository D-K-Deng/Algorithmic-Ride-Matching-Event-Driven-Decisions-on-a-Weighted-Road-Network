import json
import csv
from collections import defaultdict
import heapq
import time
import math
from datetime import datetime
from datetime import timedelta
import random

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
        
    def build_kdtree(self):
        points_with_ids = [((self.node_coordinates[node_id]['lat'], self.node_coordinates[node_id]['lon']), node_id) for node_id in self.node_coordinates]
        self.kdtree = KDTree(points_with_ids)
        
    def find_nearest_node(self, query_point):
        nearest_node = self.kdtree.find_nearest(self.kdtree.root, query_point)
        return nearest_node.id if nearest_node else None
        
        
        
class Node:
    def __init__(self, point, id, left=None, right=None):
        self.point = point  # The point (latitude, longitude)
        self.id = id        # The ID of the node
        self.left = left    # Left child
        self.right = right  # Right child

class KDTree:
    def __init__(self, points):
        self.root = self.build_tree(points)

    def build_tree(self, points, depth=0):
        if not points:
            return None

        axis = depth % 2  # 0 for latitude, 1 for longitude

        points.sort(key=lambda x: x[0][axis])  # Sort based on axis
        median = len(points) // 2

        node = Node(point=points[median][0], id=points[median][1])
        node.left = self.build_tree(points[:median], depth + 1)
        node.right = self.build_tree(points[median + 1:], depth + 1)
        return node

    def find_nearest(self, root, point, depth=0, best=None):
        if root is None:
            return best

        axis = depth % 2

        next_branch = root.left if point[axis] < root.point[axis] else root.right
        opposite_branch = root.right if point[axis] < root.point[axis] else root.left

        if best is None or self.distance(point, root.point) < self.distance(point, best.point):
            best = root

        # Recursive call to next branch
        best = self.find_nearest(next_branch, point, depth + 1, best)

        # Check if we need to explore the opposite branch
        if (point[axis] - root.point[axis]) ** 2 < self.distance(point, best.point):
            best = self.find_nearest(opposite_branch, point, depth + 1, best)

        return best

    @staticmethod
    def distance(point1, point2):
        return (point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2


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

   

def dijkstra_time_estimate(graph, start_node, end_node, date_time):
    time_of_day = format_time_of_day(date_time)
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


def dijkstra(graph, start_node):
    # Initialize distances with 'infinity' for all nodes
    distances = {node: float('infinity') for node in graph.node_coordinates}
    distances[start_node] = 0
    queue = [(0, start_node)]

    while queue:
        current_distance, current_node = heapq.heappop(queue)

        # Check if current_node is in graph.edges to avoid KeyError
        if current_node in graph.edges:
            for neighbor, edge_info in graph.edges[current_node].items():
                distance = current_distance + edge_info['length']  # edge_info may contain other data like speed
                if distance < distances[neighbor]:
                    distances[neighbor] = distance
                    heapq.heappush(queue, (distance, neighbor))

    return distances




def precompute_landmark_distances(graph, landmarks):
    landmark_distances = {}
    for landmark in landmarks:
        distances = dijkstra(graph, landmark)
        landmark_distances[landmark] = distances
    return landmark_distances


def select_landmarks(graph, number_of_landmarks):
    all_nodes = list(graph.node_coordinates.keys())
    return random.sample(all_nodes, number_of_landmarks)


def alt_heuristic(node, end_node, landmarks, landmark_distances):
    heuristic_value = 0
    for landmark in landmarks:
        # Triangle inequality: d(node, goal) ≥ |d(landmark, goal) - d(landmark, node)|
        dist_to_landmark = landmark_distances[landmark][node]
        dist_landmark_to_goal = landmark_distances[landmark][end_node]
        heuristic_value = max(heuristic_value, abs(dist_landmark_to_goal - dist_to_landmark))
    return heuristic_value


def a_star_algorithm_alt(graph, start_node, end_node, date_time, landmarks, landmark_distances):
    time_of_day = format_time_of_day(date_time)
    
    came_from = {}
    
    def get_travel_time(from_node, to_node, time_of_day):
        edge_data = graph.edges[from_node][to_node]
        distance = edge_data['length']
        speed = edge_data['speed'][time_of_day] if time_of_day in edge_data['speed'] else edge_data['speed']['default']
        return distance / max(speed, 1)  # Avoid division by zero
    
    def heuristic(node):
        heuristic_value = 0
        for landmark in landmarks:
            dist_to_landmark = landmark_distances[landmark][node]
            dist_landmark_to_goal = landmark_distances[landmark][end_node]
            heuristic_value = max(heuristic_value, abs(dist_landmark_to_goal - dist_to_landmark))
        return heuristic_value

    open_set = set([start_node])
    g_score = {node: float('infinity') for node in graph.node_coordinates}
    g_score[start_node] = 0
    f_score = {node: float('infinity') for node in graph.node_coordinates}
    f_score[start_node] = heuristic(start_node)

    while open_set:
        current_node = min(open_set, key=lambda node: f_score[node])
        if current_node == end_node:
            return reconstruct_path(came_from, end_node), g_score[end_node]  # Include total travel time

        open_set.remove(current_node)

        for neighbor, edge_info in graph.edges[current_node].items():
            tentative_g_score = g_score[current_node] + get_travel_time(current_node, neighbor, time_of_day)  # edge_info may contain other data like speed
            if tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current_node
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = tentative_g_score + heuristic(neighbor)
                open_set.add(neighbor)

    return None

def reconstruct_path(came_from, end_node):
    path = []
    current_node = end_node
    while current_node in came_from:  # Ensuring current_node is in came_from
        path.append(current_node)
        current_node = came_from[current_node]
    path.reverse()  # Reverse the path to start from the beginning
    return path



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

    return time_to_pickup + time_to_dropoff

# Improved Estimate Time Algorithm Class
class ALT_Astar_Triangle_EstimateTimeAlgorithm:
    def __init__(self, graph, landmarks, landmark_distances):
        self.graph = graph
        self.landmarks = landmarks
        self.landmark_distances = landmark_distances
        self.available_drivers = []
        self.unmatched_passengers = []
        self.driver_heap = []
        self.total_matches = 0
        self.total_travel_time = 0
        self.total_time_to_pickup =0
        self.total_time_to_dropoff=0

    def add_driver(self, time, node):
        heapq.heappush(self.available_drivers, (time, node))

    def add_passenger(self, time, pickup_node, dropoff_node):
        heapq.heappush(self.unmatched_passengers, (time, pickup_node, dropoff_node))

    def initialize_heap(self):
        # Called once to initialize the heap
        self.driver_heap = [(driver[0], driver[1]) for driver in self.available_drivers]
        heapq.heapify(self.driver_heap)

    def match_passenger_to_driver(self):
        if self.driver_heap and self.unmatched_passengers:
            earliest_available_time, driver_location = heapq.heappop(self.driver_heap)
            
            # Match with the first passenger in the queue
            passenger = self.unmatched_passengers.pop(0)
            pickup_location, dropoff_location = passenger[1], passenger[2]

            _, travel_time_to_pickup = a_star_algorithm_alt(self.graph, driver_location, pickup_location, earliest_available_time,self.landmarks, self.landmark_distances)
            _, travel_time_to_dropoff = a_star_algorithm_alt(self.graph, pickup_location, dropoff_location, earliest_available_time + timedelta(seconds=travel_time_to_pickup),self.landmarks, self.landmark_distances)
            
            total_travel_time = travel_time_to_pickup + travel_time_to_dropoff
            new_available_time = earliest_available_time + timedelta(seconds=total_travel_time)

            # Push the driver back into the heap with updated availability
            heapq.heappush(self.driver_heap, (new_available_time, dropoff_location))

            # Update total matches and travel time
            self.total_matches += 1
            self.total_travel_time += total_travel_time
            self.total_time_to_pickup+=travel_time_to_pickup
            self.total_time_to_dropoff+=travel_time_to_dropoff
            
            return total_travel_time

        return None

    
    def report_stats(self):
        print(f"Total matches made: {self.total_matches}")
        print(f"Total travel time for all matches: {self.total_travel_time}")
        print(f"Total time to pick up:{self.total_time_to_pickup}")
        print(f"Total time to drop off:{self.total_time_to_dropoff} ")
        print(f"Passengers' total travel time:{self.total_time_to_pickup+self.total_time_to_dropoff}")
        print(f"Drivers' profit:{self.total_time_to_dropoff-self.total_time_to_pickup}")





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

def load_drivers_kdtree(filename, graph):
    drivers = []
    with open(filename, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            date_time = datetime.strptime(row['Date/Time'], "%m/%d/%Y %H:%M:%S")
            source_lat = float(row['Source Lat'])
            source_lon = float(row['Source Lon'])
            nearest_node = graph.find_nearest_node((source_lat, source_lon))
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


def load_passengers_kdtree(filename, graph):
    passengers = []
    with open(filename, newline='') as file:
        reader = csv.DictReader(file)
        for row in reader:
            date_time = datetime.strptime(row['Date/Time'], "%m/%d/%Y %H:%M:%S")
            source_lat = float(row['Source Lat'])
            source_lon = float(row['Source Lon'])
            dest_lat = float(row['Dest Lat'])
            dest_lon = float(row['Dest Lon'])
            source_node = graph.find_nearest_node((source_lat, source_lon))
            dest_node = graph.find_nearest_node((dest_lat, dest_lon))
            passengers.append((date_time, source_node, dest_node))
    return passengers

    
def T5():
    start_time = time.time()
    
    
    
    graph = load_edge_data("edges.csv")
    load_node_data("node_data.json",graph)
    
    num_landmarks = 100
    
    landmarks = select_landmarks(graph, num_landmarks)
    
    landmark_distances = precompute_landmark_distances(graph, landmarks)
    
    algorithm = ALT_Astar_Triangle_EstimateTimeAlgorithm(graph, landmarks, landmark_distances)
    
    graph.build_kdtree()
    
    drivers = load_drivers_kdtree("drivers.csv", graph)
    for driver_time, driver_node in drivers:
        algorithm.add_driver(driver_time, driver_node)
        
    passengers = load_passengers_kdtree("passengers.csv", graph)
    for passenger in passengers:
        algorithm.add_passenger(*passenger)
    
    
    algorithm.initialize_heap()
    print("pre-process finished")

    while algorithm.unmatched_passengers:
        travel_time = algorithm.match_passenger_to_driver()


    algorithm.report_stats()

    
    end_time = time.time()
    print(f"Program running time: {end_time - start_time} seconds")    
    

    
T5()