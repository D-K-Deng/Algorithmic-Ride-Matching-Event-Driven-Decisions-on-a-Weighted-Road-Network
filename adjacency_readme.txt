adjacency.json

adjacency.json contains an adjacency list dictionary that represents the connections between different nodes, along with associated attributes for each connection.

The dictionary has the following structure:
{ 
    start_node_id <ID of the start node, int>: {
        end_node_id <ID of the end node, int>: {
            'day_type': <type of day, string, 'weekday'/'weekend'>,
            'hour': <hour of the day, int, 0-23>,
            'length': <length of the connection, int, converted to miles>,
            'max_speed': <maximum speed, float, converted to miles per hour>,
            'time': <length/max_speed, float, converted to hours>
        },
        ...
    },
    ...
}