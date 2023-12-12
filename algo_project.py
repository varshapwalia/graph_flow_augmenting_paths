import random
import heapq
import csv

class Vertex:
    def __init__(self, name, x, y):
        self.vertex_name=name
        self.x = x
        self.y = y
        self.edges=[]
    
    def __str__(self):
        return f"{self.vertex_name}"
        

class Edge:
    def __init__(self, u, v, capacity):
        self.u = u
        self.v = v
        self.capacity = capacity
        self.flow = 0

class Graph:
    def __init__(self):
        self.vertices = []
        self.edges = []

    def get_capacity(self, u, v):
        for edge in self.edges:
            if edge.u == u and edge.v == v:
                return edge.capacity
        return 0

    def update_flow(self, u, v, min_residual_capacity):
        for edge in self.edges:
            if edge.u == u and edge.v == v:
                edge.flow += min_residual_capacity
                return
    
    def get_flow(self, u, v):
        for edge in self.edges:
            if edge.u == u and edge.v == v:
                return edge.flow
        return 0

def generate_sink_source_graph(n, r, upper_cap):
    graph = Graph()

    # Step 1: Define a set of vertices V such that |V| = n
    vertex_num=0
    for _ in range(n):
        vertex_num+=1
        # print(vertex_num)
        graph.vertices.append(Vertex(vertex_num, random.uniform(0, 1), random.uniform(0, 1)))

    # Step 2: Randomly assign edges of length ≤ r without creating parallel edges
    for u in graph.vertices:
        for v in graph.vertices:
            if u != v and (u.x - v.x)**2 + (u.y - v.y)**2 <= r**2:
                if (not any(edge.u == u and edge.v == v for edge in graph.edges)) and (not any(edge.u == v and edge.v == u for edge in graph.edges)):
                    rand = random.uniform(0, 1)
                    if rand < 0.5:
                        edge = Edge(u, v, random.randint(1, upper_cap)) # Step 3: Assign randomly selected integer-value capacity in the range [1..upperCap]
                        graph.edges.append(edge)
                        u.edges.append(edge)
                    else:
                        edge = Edge(v, u, random.randint(1, upper_cap))
                        graph.edges.append(edge)
                        v.edges.append(edge)

    # Step 4: Randomly select one of the nodes as the source node s
    source = random.choice(graph.vertices)

    # Step 5: Apply BFS to find a longest acyclic path and define the end node of this path as the sink t
    visited = set()
    queue = [(source, [])]

    while queue:
        current, path = queue.pop(0)
        if current not in visited:
            visited.add(current)
            path = path + [current]

            for edge in current.edges:
                neighbor = edge.v
                if neighbor not in visited:
                    queue.append((neighbor, path))

    sink = path[-1]

    return graph, source, sink

# Shortest augmenting path

def initialize_single_source(graph, source):
        dist = {vertex: float('inf') for vertex in graph.vertices}
        dist[source] = 0
        return dist

def relax(u, v, w, dist, prev):
        if dist[v] > dist[u] + w[u][v]:
            dist[v] = dist[u] + w[u][v]
            prev[v] = u

def shortest_augmenting_path(graph, source, sink):
    
    # Initialize distances and predecessors
    dist = initialize_single_source(graph, source)
    prev = {vertex: None for vertex in graph.vertices}
    visited = set()

    # Priority queue for efficient minimum extraction
    priority_queue = []
    heapq.heappush(priority_queue, (0,0,source))
    counter=0

    while priority_queue:
        _,_, u = heapq.heappop(priority_queue)

        if u in visited:
            continue

        visited.add(u)

        # Iterate over outgoing edges from vertex u
        for edge in u.edges:
            counter=counter+1 # we take counter because if the key is same then heap try to compare next value in tuple which will be incident vertice enounter
            v = edge.v
            if edge.flow < edge.capacity:
                # Relax the edge and update the priority queue
                weight_uv = 1 # unit weight
                relax(u, v, {u: {v: weight_uv}}, dist, prev)
                heapq.heappush(priority_queue, (dist[v], counter, v)) #counter will come in picture if we have 2 or more same 


    # Reconstruct the augmenting path from source to sink
    augmenting_path = []
    current = sink

    while prev[current] is not None:
        augmenting_path.append((prev[current], current))
        current = prev[current]

    augmenting_path.reverse()
    return augmenting_path

# DFS_like

def dfs_like_initialize_single_source(graph, source):
        dist = {vertex: float('inf') for vertex in graph.vertices}
        dist[source] = 0
        return dist

def dfs_like_relax(u, v, w, dist, prev):
        if dist[v] == float('inf'):
            dist[v] = w[u][v]
            prev[v] = u

def dfs_like(graph, source, sink):
    
    # Initialize distances and predecessors
    dist = dfs_like_initialize_single_source(graph, source)
    prev = {vertex: None for vertex in graph.vertices}
    visited = set()

    # Priority queue for efficient minimum extraction
    priority_queue = []
    heapq.heappush(priority_queue, (0,0,source))
    counter=9999999

    while priority_queue:
        _,_, u = heapq.heappop(priority_queue)

        if u in visited:
            continue

        visited.add(u)

        # Iterate over outgoing edges from vertex u
        for edge in u.edges:
            counter=counter-1 # we take counter because if the key is same then heap try to compare next value in tuple which will be incident vertice enounter
            v = edge.v
            if edge.flow < edge.capacity:
                # Relax the edge and update the priority queue
                dfs_like_relax(u, v, {u: {v: counter}}, dist, prev)
                heapq.heappush(priority_queue, (dist[v], counter, v))


    # Reconstruct the augmenting path from source to sink
    augmenting_path = []
    current = sink

    while prev[current] is not None:
        augmenting_path.append((prev[current], current))
        current = prev[current]

    augmenting_path.reverse()
    return augmenting_path

# Max-Cap

def max_cap_initialize_single_source(graph, source):
        dist = {vertex: float('inf') for vertex in graph.vertices}
        dist[source] = 0
        return dist

def max_cap_relax(u, v, w, dist, prev):
        if dist[v] > dist[u] + w[u][v]:
            dist[v] = dist[u] + w[u][v]
            prev[v] = u

def max_cap(graph, source, sink):
    
    # Initialize distances and predecessors
    dist = max_cap_initialize_single_source(graph, source)
    prev = {vertex: None for vertex in graph.vertices}
    visited = set()

    # Priority queue for efficient minimum extraction
    priority_queue = []
    heapq.heappush(priority_queue, (0,0,source))
    counter=0

    while priority_queue:
        _,_, u = heapq.heappop(priority_queue)

        if u in visited:
            continue

        visited.add(u)

        # Iterate over outgoing edges from vertex u
        for edge in u.edges:
            counter=counter+1 # we take counter because if the key is same then heap try to compare next value in tuple which will be incident vertice enounter
            v = edge.v
            if edge.flow < edge.capacity:
                # Relax the edge and update the priority queue
                max_cap_relax(u, v, {u: {v: edge.capacity}}, dist, prev)
                heapq.heappush(priority_queue, (-dist[v], counter, v)) # we are putting distaces are as negative to get max element from Q


    # Reconstruct the augmenting path from source to sink
    augmenting_path = []
    current = sink

    while prev[current] is not None:
        augmenting_path.append((prev[current], current))
        current = prev[current]

    augmenting_path.reverse()
    return augmenting_path

# random path
def random_initialize_single_source(graph, source):
        dist = {vertex: float('inf') for vertex in graph.vertices}
        dist[source] = 0
        return dist

def random_relax(u, v, w, dist, prev):
        if dist[v] == float('inf'):
            dist[v] = w[u][v]
            prev[v] = u

def generate_unique_random_number(existing_numbers, lower_limit, upper_limit):
    while True:
        random_number = random.randint(lower_limit, upper_limit)
        if random_number not in existing_numbers:
            existing_numbers.append(random_number)
            return random_number

def random_path(graph, source, sink):
    
    # Initialize distances and predecessors
    dist = random_initialize_single_source(graph, source)
    prev = {vertex: None for vertex in graph.vertices}
    visited = set()

    # Priority queue for efficient minimum extraction
    priority_queue = []
    heapq.heappush(priority_queue, (0,0,source))

    existing_numbers_set = [0]
    

    while priority_queue:
        _,_, u = heapq.heappop(priority_queue)

        if u in visited:
            continue

        visited.add(u)

        # Iterate over outgoing edges from vertex u
        for edge in u.edges:
            
            random_number = generate_unique_random_number(existing_numbers_set, 1, 99999) # we take counter because if the key is same then heap try to compare next value in tuple which will be incident vertice enounter
            v = edge.v
            if edge.flow < edge.capacity:
                # Relax the edge and update the priority queue
                random_relax(u, v, {u: {v: random_number}}, dist, prev)
                heapq.heappush(priority_queue, (dist[v], random_number, v))


    # Reconstruct the augmenting path from source to sink
    augmenting_path = []
    current = sink

    while prev[current] is not None:
        augmenting_path.append((prev[current], current))
        current = prev[current]

    augmenting_path.reverse()
    return augmenting_path

###
# paths: the number of augmenting paths required until Ford-Fulkerson completes
# mean length (ML): average length (i.e., number of edges) of the augmenting paths
# mean proportional length (MPL): the average length of the augmenting path as a fraction ofthe longest acyclic path from s to t
# total edges: the total number of edges in the graph
###

def ford_fulkerson_method(graph, source, sink, augmenting_path_algorithm):

    paths = 0
    total_length = 0
    max_length = 0
    total_edges = len(graph.edges)

    while True:
        augmenting_path = augmenting_path_algorithm(graph, source, sink)

        if not augmenting_path:
            break

        paths += 1
        length = len(augmenting_path)
        total_length += length
        max_length = max(max_length, length)

        # Find the minimum capacity along the augmenting path
        min_residual_capacity = min(graph.get_capacity(u, v)-graph.get_flow(u, v) for u, v in augmenting_path)

        # Update the residual graph with the flow along the augmenting path
        for u, v in augmenting_path:
            graph.update_flow(u, v, min_residual_capacity)


    mean_length = total_length / paths if paths > 0 else 0
    mean_proportional_length = mean_length / max_length if max_length > 0 else 0

    return paths, round(mean_length, 4), round(mean_proportional_length, 4), total_edges

def save_graph_to_csv(graph, source, sink, filename):
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)

        # Write vertices
        writer.writerow(['VertexName', 'X', 'Y'])
        for vertex in graph.vertices:
            writer.writerow([vertex.vertex_name, vertex.x, vertex.y])

        writer.writerow([])  # Blank line to separate sections

        # Write edges
        writer.writerow(['From', 'To', 'Capacity', 'Flow'])
        for edge in graph.edges:
            writer.writerow([edge.u.vertex_name, edge.v.vertex_name, edge.capacity, edge.flow])

        writer.writerow([])  # Blank line to separate sections

        # Write source and sink
        writer.writerow(['Source'])
        writer.writerow([source.vertex_name])

        writer.writerow([])  # Blank line to separate sections

        writer.writerow(['Sink'])
        writer.writerow([sink.vertex_name])

def load_graph_from_csv(filename):
    graph = Graph()
    vertices = []
    edges = []
    source = None
    sink = None

    with open(filename, 'r') as csvfile:
        reader = csv.reader(csvfile)

        section = ""
        for row in reader:
            if not row:
                continue

            if row[0] == "VertexName":
                section = "Vertices"
                continue
            elif row[0] == "From":
                section = "Edges"
                continue
            elif row[0] == "Source":
                section = "Source"
                continue
            elif row[0] == "Sink":
                section = "Sink"
                continue

            if section == "Vertices":
                vertex_name, x, y = map(float, row)
                vertex = Vertex(vertex_name, x, y)
                vertices.append(vertex)

            elif section == "Edges":
                from_vertex, to_vertex, capacity, _ = map(float, row)
                u = next((vertex for vertex in vertices if vertex.vertex_name == from_vertex), None)
                v = next((vertex for vertex in vertices if vertex.vertex_name == to_vertex), None)
                if u and v:
                    edge = Edge(u, v, int(capacity))
                    edge.flow = 0
                    edges.append(edge)
                    u.edges.append(edge)

            elif section == "Source":
                source_name = float(row[0])
                source = next((vertex for vertex in vertices if vertex.vertex_name == source_name), None)

            elif section == "Sink":
                sink_name = float(row[0])
                sink = next((vertex for vertex in vertices if vertex.vertex_name == sink_name), None)

    graph.vertices = vertices
    graph.edges = edges

    if source is None or sink is None:
        raise ValueError("Source or sink not found in the loaded vertices.")

    return graph, source, sink


# # Simulations I
values_i = [
    (100, 0.2, 2),
    (200, 0.2, 2),
    (100, 0.3, 2),
    (200, 0.3, 2),
    (100, 0.2, 50),
    (200, 0.2, 50),
    (100, 0.3, 50),
    (200, 0.3, 50),
]

# Commented becuase graph Generated
# for n, r, upper_cap in values_i:
#     graph, source, sink = generate_sink_source_graph(n, r, upper_cap)
#     save_graph_to_csv(graph, source, sink, f"Simulation_1_Graph_{n}_{r}_{upper_cap}.csv")

# Print the results in a table format
print("\nSimulations I Results:")
print("Algorithm n r upperCap paths ML MPL total edges")

for n, r, upper_cap in values_i:

    graph, source, sink = load_graph_from_csv(f"Simulation_1_Graph_{n}_{r}_{upper_cap}.csv")
    sap_result = ford_fulkerson_method(graph, source, sink, shortest_augmenting_path)
    print("SAP", n, r, upper_cap, sap_result)

    graph, source, sink = load_graph_from_csv(f"Simulation_1_Graph_{n}_{r}_{upper_cap}.csv")
    dfs_result=ford_fulkerson_method(graph, source, sink, dfs_like)
    print("DFS", n, r, upper_cap, dfs_result)

    graph, source, sink = load_graph_from_csv(f"Simulation_1_Graph_{n}_{r}_{upper_cap}.csv")
    max_cap_result=ford_fulkerson_method(graph, source, sink, max_cap)
    print("MaxCap", n, r, upper_cap, max_cap_result)

    graph, source, sink = load_graph_from_csv(f"Simulation_1_Graph_{n}_{r}_{upper_cap}.csv")
    random_path_result=ford_fulkerson_method(graph, source, sink, random_path)
    print("Random", n, r, upper_cap, random_path_result)

# # Simulations II
values_ii = [
    (100, 0.2, 5),
    (200, 0.3, 50),
    (150, 0.15, 25),
]

# Commented becuase graph Generated
# Perform Simulations II
# for n, r, upper_cap in values_ii:
#     graph, source, sink = generate_sink_source_graph(n, r, upper_cap)
#     save_graph_to_csv(graph, source, sink, f"Simulation_2_Graph_{n}_{r}_{upper_cap}.csv")

# Print the results in a table format
print("\nSimulations II Results:")
print("Algorithm n r upperCap paths ML MPL total edges")

for n, r, upper_cap in values_ii:
    
    graph, source, sink = load_graph_from_csv(f"Simulation_2_Graph_{n}_{r}_{upper_cap}.csv")
    sap_result = ford_fulkerson_method(graph, source, sink, shortest_augmenting_path)
    print("SAP", n, r, upper_cap, sap_result)

    graph, source, sink = load_graph_from_csv(f"Simulation_2_Graph_{n}_{r}_{upper_cap}.csv")
    dfs_result = ford_fulkerson_method(graph, source, sink, dfs_like)
    print("DFS", n, r, upper_cap, dfs_result)

    graph, source, sink = load_graph_from_csv(f"Simulation_2_Graph_{n}_{r}_{upper_cap}.csv")
    max_cap_result = ford_fulkerson_method(graph, source, sink, max_cap)
    print("MaxCap", n, r, upper_cap, max_cap_result)

    graph, source, sink = load_graph_from_csv(f"Simulation_2_Graph_{n}_{r}_{upper_cap}.csv")
    random_path_result = ford_fulkerson_method(graph, source, sink, random_path)
    print("Random", n, r, upper_cap, random_path_result)

# Commented becuase graph Generated
#Code for Implmentation correctness to check is all give results for a small graph
# graph, source, sink = generate_sink_source_graph(20, 0.2, 10)
# save_graph_to_csv(graph, source, sink, f"Implementation_correctness_Test_Graph_{20}_{0.2}_{10}.csv")

print("\nTesting Results:")
print("Algorithm n r upperCap paths ML MPL total edges")

graph, source, sink = load_graph_from_csv(f"Implementation_correctness_Test_Graph_{20}_{0.2}_{10}.csv")
sap_result = ford_fulkerson_method(graph, source, sink, shortest_augmenting_path)
print("SAP", 20, 0.2, 10, sap_result)

graph, source, sink = load_graph_from_csv(f"Implementation_correctness_Test_Graph_{20}_{0.2}_{10}.csv")
dfs_result = ford_fulkerson_method(graph, source, sink, dfs_like)
print("DFS", 20, 0.2, 10, dfs_result)

graph, source, sink = load_graph_from_csv(f"Implementation_correctness_Test_Graph_{20}_{0.2}_{10}.csv")
max_cap_result = ford_fulkerson_method(graph, source, sink, max_cap)
print("MaxCap", 20, 0.2, 10, max_cap_result)

graph, source, sink = load_graph_from_csv(f"Implementation_correctness_Test_Graph_{20}_{0.2}_{10}.csv")
random_path_result = ford_fulkerson_method(graph, source, sink, random_path)
print("Random", 20, 0.2, 10, random_path_result)