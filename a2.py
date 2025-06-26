"""
FIT2004 

Name: Ooi Hui Xia
-----------------------------------------------------------------------------------------------
"""



from math import inf  

"""
-----------------------------------------------------------------------------------------------
Question 1
-----------------------------------------------------------------------------------------------
"""

class Edge:
    """
    Class Description:
    Represents a directed edge in the residual graph with capacity and current flow.
    
    Approach Description:
    Models the residual graph edge that allows augmentation of flow. Each edge maintains a reference to its reverse edge to easily update reverse flows during augmentation steps. 
    This enables efficient residual capacity updates in the Ford-Fulkerson method.
    
    Input:
    - from_node (int): Starting vertex index of the edge.
    - to_node (int): Ending vertex index.
    - capacity (int): Maximum flow capacity.
    - reverse_index (int): Index of the reverse edge.
    
    Output: None
    
    Time Complexity: O(1) for initialization and for remaining_capacity method
    
    Time Complexity Analysis: 
    Edge creation and residual capacity calculation are constant-time operations.

    Space Complexity: O(1) per edge

    Space Complexity Analysis: Stores fixed number of integers and a flow value per edge.
    """
    def __init__(self, from_node, to_node, capacity, reverse_index):
        self.from_node = from_node # The starting node of the edge
        self.to_node = to_node # The ending node of the edge
        self.capacity = capacity # Max capacity of the edge 
        self.flow = 0 # How much flow is currently used 
        self.reverse_index = reverse_index # Index of the reverse edge in the adjacency list 
    
    def remaining_capacity(self):
        """
        Returns how much flow can still be pushed through this edge.

        The value is calculated as capacity - flow and is used during augmenting path discovery.

        Time and Space complexities: O(1)
        """
        return self.capacity - self.flow

class Vertex:
    """
    Class Description: 
    Represents a vertex in the residual graph used for BFS during the Ford-Fulkerson algorithm.
    Track visitation state and the edge used to reach this vertex.
    
    Approach Description: 
    Used to keep track of visited status and path (previous edge) during BFS.
    Mark if the vertex has been visited.
    Remember which edge was used to reach this vertex (helps in reconstructing the path)
    
    Input: index (int): The vertex index in the graph
    
    Output: None
    
    Time Complexity: O(1) 

    Time Complexity Analysis: Each vertex stores fixed state information for BFS traversal.
    
    Space Complexity: O(1) per vertex

    Space Complexity Analysis: Only fixed attributes for tracking BFS visitation and path.
    """
    def __init__(self, index):
        self.index = index
        self.visited = False
        self.previous_edge = None # The edge used to reach this node during BFS

class ResidualGraph:
    """
    Class Description:
    Represents a residual graph for use in the Ford-Fulkerson max flow algorithm.
    Maintains an adjacency list of directed edges with capacities and flows. 
    
    Approach Description:
    Models a flow network where each edge tracks has a capacity and current flow.
    It maintains an adjacency list of edges and provides methods to add edges and find augmenting paths using BFS.
    The residual graph allows flow to be "pushed" along edges and reversed if needed, which is essential for updating flow in Ford-Fulkerson.
        
    Solution Description:
    The graph is represented as a list of adjacency lists, each storing Edge objects.
    When adding an edge, a forward edge with capacity and a backward edge with zero capacity are created to represent residual capacity.
    BFS is used to find paths with available capacity from source to sink. Once an augmenting path is found, the bottleneck capacity (minimum residual capacity on that path) is computed, and flow is augmented accordingly.
    These operations enable incrementally increasing flow until no augmenting path exists.


    Input: num_nodes (int): Total number of nodes in the residual graph
    
    Output: None
    
    Time Complexity: 
    - add_edge: O(1) per edge.
    - bfs_find_augmenting_path: O(V + E) per BFS, where V is vertices, E edges. 
    - find_bottleneck_capacity: O(V) to retrace path. 
    - augment_flow_along_path: O(V) to update flow.
    Overall per augmenting path: O(V + E).


    Space Complexity:
    - Stores adjacency lists for all nodes: O(V + E)
    - Each edge stored twice (forward and backward).
    Overall space: O(V + E).

    
    Space Complexity Analysis:
    Adjacency lists allow efficient traversal of neighbors. Each vertex and edge is visited at most once during BFS. Augmenting path tracking requires constant memory per vertex. 
    This supports efficient updates during each flow augmentation step.
    """
    def __init__(self, num_nodes):
        self.adjacency_list = [[] for _ in range(num_nodes)]
        self.num_nodes = num_nodes
    
    def add_edge(self, from_node, to_node, capacity):
        forward_edge = Edge(from_node, to_node, capacity, len(self.adjacency_list[to_node]))
        backward_edge = Edge(to_node, from_node, 0, len(self.adjacency_list[from_node]))
        self.adjacency_list[from_node].append(forward_edge)
        self.adjacency_list[to_node].append(backward_edge)
    
    def bfs_find_augmenting_path(self, source, sink, vertices):
        # Reset BFS state 
        for vertex in vertices:
            vertex.visited = False
            vertex.previous_edge = None
        
        queue = [vertices[source]]
        vertices[source].visited = True
        front = 0
        
        while front < len(queue):
            current_vertex = queue[front]
            front += 1
            
            for edge in self.adjacency_list[current_vertex.index]:
                if not vertices[edge.to_node].visited and edge.remaining_capacity() > 0:
                    vertices[edge.to_node].visited = True
                    vertices[edge.to_node].previous_edge = edge
                    if edge.to_node == sink:
                        return True
                    queue.append(vertices[edge.to_node])
        return False
    
    def find_bottleneck_capacity(self, source, sink, vertices):
        # Walk backward from sink to source using previous_edge to find minimum residual capacity
        path_flow = inf
        current = sink
        while current != source:
            edge = vertices[current].previous_edge
            path_flow = min(path_flow, edge.remaining_capacity())
            current = edge.from_node
        return path_flow
    
    def augment_flow_along_path(self, source, sink, vertices, path_flow):
        current = sink
        while current != source:
            edge = vertices[current].previous_edge
            edge.flow += path_flow
            reverse_edge = self.adjacency_list[edge.to_node][edge.reverse_index]
            reverse_edge.flow -= path_flow
            current = edge.from_node

def ford_fulkerson_max_flow(graph, source, sink, vertices):
    """
    Function Description:
    Computes the maximum flow from the source to the sink in a given residual graph using the Ford-Fulkerson method with BFS to find augmenting paths.

    Approach Description:
    This function implements the Ford-Fulkerson algorithm using the BFS variant known as the Edmonds-Karp algorithm. It repeatedly finds an augmenting path from source to sink using BFS, calculates the bottleneck capacity of that path, 
    and augments the flow along it. This process continues until no more augmenting paths are found.

    Solution Description:
    - Initialize max_flow to 0.
    - While BFS finds a valid augmenting path:
    1. Determine the minimum residual capacity (bottleneck) along that path.
    2. Increase flow along the path by this bottleneck amount.
    3. Update the reverse edges accordingly.
    - Once no augmenting path exists, return the accumulated max_flow.

    Inputs:
    - graph (ResidualGraph): The graph containing vertices and edges with flow and capacity.
    - source (int): Index of the source vertex.
    - sink (int): Index of the sink vertex.
    - vertices (List[Vertex]): List of Vertex objects representing each node.

    Output:
    - max_flow (int): The maximum amount of flow that can be sent from source to sink.

    Time Complexity:
    - Each BFS: O(V + E), where V is number of vertices, E is number of edges.
    - Number of iterations (augmenting paths found): O(E * max_flow), in worst case.
    - Total: O(E * max_flow) using simple Ford-Fulkerson.
    With BFS (Edmonds-Karp): O(V * E²), since each BFS takes O(E), and there are O(VE) iterations.

    Space Complexity: O(V + E) for graph and BFS traversal structures.

    Complexity Analysis:
    - Each call to BFS traverses the graph using adjacency lists, visiting each edge once.
    - The number of times an edge can be part of a BFS is proportional to the number of times flow is augmented before the edge becomes saturated.
    - The flow increases by at least 1 unit per iteration (in integer capacity graphs), so the total number of iterations is bounded by the maximum possible flow.
    """
    max_flow = 0
    while True:
        # for vertex in vertices:
            #vertex.visited = False
            #vertex.previous_edge = None
        if not graph.bfs_find_augmenting_path(source, sink, vertices):
            break
        flow = graph.find_bottleneck_capacity(source, sink, vertices)
        graph.augment_flow_along_path(source, sink, vertices, flow)
        max_flow += flow
    return max_flow

class Graph:
    """
    Function description:  
    This class models the problem of assigning students to classes based on their time preferences and class capacity constraints. It uses a flow network with circulation and applies the 
    Ford-Fulkerson algorithm to check feasibility and compute valid assignments.

    Approach description:  
    The problem is modeled as a flow network:
    - Each student connects to the source with capacity 1 (each can join one class).
    - Each class connects to the sink with min/max student limits.
    - Students are connected to classes only if the class's time matches one of their preferences.

    To handle minimum class sizes, we use circulation: demands and a super source/sink are added to ensure constraints are respected. Lower/upper bounds on edges are encoded as adjustments to demand and capacity.

    I first check if a feasible flow exists (circulation check). If it does, it computes a flow from the original source to sink to get the actual allocation. 
    This avoids brute force or greedy logic — the assignment satisfies constraints, and multiple valid outputs may exist.

    Input:  
    - num_students: int — number of students  
    - num_classes: int — number of classes  
    - time_preferences: List[List[int]] — each student's preferred time slots  
    - class_constraints: List[List[int]] — for each class: [time_slot, min_students, max_students]  

    Output / Postcondition:  
    - get_allocation(): list[int] — for each student, the index of the assigned class, or -1 if unassigned  
    - check_feasibility(): bool — True if an assignment satisfying all constraints exists

    Time complexity:  O(n²)

    Time complexity analysis:  
    The number of edges is O(n²) in the worst case (each student prefers all classes).  
    Ford-Fulkerson on unit capacity graphs runs in O(EF), where F is the maximum flow (≤ n), and E is O(n^2).  
    Since each augmenting path increases flow by 1 and DFS finds each in O(n²), but with unit capacities and bounded flow, it use at most n augmentations, each costing O(n²), so total is O(n²).

    Space complexity:  O(n)

    Space complexity analysis: 
    Auxiliary arrays and temporary objects (e.g., demand[], visited[], vertex states) take O(n) space.  
    Graph storage is excluded from auxiliary space per problem definition.
    """
    def __init__(self, num_students, num_classes, time_preferences, class_constraints):
        self.num_students = num_students
        self.num_classes = num_classes
        self.time_preferences = time_preferences
        self.class_constraints = class_constraints

        self.S = 0
        self.T = num_students + num_classes + 1
        self.super_source = self.T + 1
        self.super_sink = self.T + 2
        self.total_nodes = self.super_sink + 1

        self.graph = ResidualGraph(self.total_nodes)
        self.demand = [0] * self.total_nodes

    def add_bounded_edge(self, from_node, to_node, lower_bound, upper_bound):
        self.demand[from_node] -= lower_bound
        self.demand[to_node] += lower_bound
        self.graph.add_edge(from_node, to_node, upper_bound - lower_bound)

    def build_network(self):
        # Source to students (1 unit each)
        for student_idx in range(self.num_students):
            self.add_bounded_edge(self.S, 1 + student_idx, 1, 1)

        # Class to sink with min/max bounds
        for class_idx in range(self.num_classes):
            class_vertex = 1 + self.num_students + class_idx
            min_students = self.class_constraints[class_idx][1]
            max_students = self.class_constraints[class_idx][2]
            self.add_bounded_edge(class_vertex, self.T, min_students, max_students)

        # Student to class (based on time preferences)
        classes_at_time = [[] for _ in range(20)]
        for class_idx, class_info in enumerate(self.class_constraints):
            time_slot = class_info[0]
            classes_at_time[time_slot].append(class_idx)

        for student_idx in range(self.num_students):
            student_vertex = 1 + student_idx
            for preferred_time in self.time_preferences[student_idx]:
                for class_idx in classes_at_time[preferred_time]:
                    class_vertex = 1 + self.num_students + class_idx
                    self.add_bounded_edge(student_vertex, class_vertex, 0, 1)

        # Super source and super sink for circulation feasibility
        self.total_positive_demand = 0
        for node in range(self.total_nodes):
            if self.demand[node] > 0:
                self.graph.add_edge(self.super_source, node, self.demand[node])
                self.total_positive_demand += self.demand[node]
            elif self.demand[node] < 0:
                self.graph.add_edge(node, self.super_sink, -self.demand[node])

        # Infinite edge from T to S to complete circulation
        self.graph.add_edge(self.T, self.S, float('inf'))

    def check_feasibility(self):
        vertices = [Vertex(i) for i in range(self.total_nodes)]
        max_flow = ford_fulkerson_max_flow(self.graph, self.super_source, self.super_sink, vertices)
        return max_flow == self.total_positive_demand

    def compute_actual_flow(self):
        # Run max flow from S to T after feasibility is confirmed
        vertices = [Vertex(i) for i in range(self.total_nodes)]
        ford_fulkerson_max_flow(self.graph, self.S, self.T, vertices)

    def get_allocation(self):
        allocation = [-1] * self.num_students
        for student_idx in range(self.num_students):
            student_vertex = 1 + student_idx
            for edge in self.graph.adjacency_list[student_vertex]:
                if (1 + self.num_students) <= edge.to_node < (1 + self.num_students + self.num_classes) and edge.flow == 1:
                    allocation[student_idx] = edge.to_node - (1 + self.num_students)
                    break
        return allocation


def crowdedCampus(n, m, timePreferences, proposedClasses, minimumSatisfaction):
    """
    Function Description:
    Assign students to classes respecting their time preferences to satisfy at least a minimum number of students using a flow network and Ford-Fulkerson max flow algorithm.

    Approach Description:
    The core of the problem is to assign students to classes such that students only get classes offered at times they prefer, and the total number of satisfied students
    meets a required minimum. Instead of enumerating all possible allocations (which is combinatorial and inefficient) or using a greedy heuristic (which may fail to find
    feasible solutions), the problem is modeled as a flow network.

    In this model:
    - Each student is represented as a node connected from a sourcenode.
    - Each class is represented as a node connected to a sink node.
    - Edges from students to classes exist only if the class time matches the student's preferred times, forming possible valid assignments.

    The capacity on each edge is typically 1, enforcing that one student can take one class.
    Finding a maximum flow through this network corresponds to assigning as many students as possible to classes they prefer.

    The Ford-Fulkerson algorithm is implemented to find a feasible maximum flow, which gives an allocation meeting the minimum satisfaction.

    Input:
    - n (int): Number of students.
    - m (int): Number of classes.
    - timePreferences (List[List[int]]): Each student's ordered list of preferred class times (a full permutation of 0 to 19).
    - proposedClasses (List[List[int]]): List of classes with their time slots.
    - minimumSatisfaction (int): Minimum number of students to satisfy.

    Output:
    - List[int]: Allocation list assigning each student to a class index, or -1 if none.
    - None if no feasible allocation meets minimum satisfaction.

    Time Complexity: O(n^2), dominated by Ford-Fulkerson max flow.

    Time Complexity Analysis: Graph size limited by preferences, max 5 edges per student initially, max flow runs in O(n^2) worst-case, possibly twice due to two attempts.

    Space Complexity: O(n) auxiliary space for graph and allocations.

    Space Complexity Analysis: Linear storage for nodes, edges, flows, and assignment arrays.
    """
    top5_prefs = []
    for prefs in timePreferences:
        top_prefs = []
        count = 0
        for p in prefs:
            if count == 5:
                break
            top_prefs.append(p)
            count += 1
        top5_prefs.append(top_prefs)

    graph = Graph(n, m, top5_prefs, proposedClasses)
    graph.build_network()
    if graph.check_feasibility():
        graph.compute_actual_flow()
        allocation = graph.get_allocation()
        satisfied = 0
        for i in range(n):
            if allocation[i] != -1:
                count = 0
                for pref in timePreferences[i]:
                    if count == 5:
                        break
                    if proposedClasses[allocation[i]][0] == pref:
                        satisfied += 1
                        break
                    count += 1
        if satisfied >= minimumSatisfaction:
            return allocation

    graph = Graph(n, m, timePreferences, proposedClasses)
    graph.build_network()
    if not graph.check_feasibility():
        return None
    graph.compute_actual_flow()
    allocation = graph.get_allocation()

    satisfied = 0
    for i in range(n):
        if allocation[i] != -1:
            count = 0
            for pref in timePreferences[i]:
                if count == 5:
                    break
                if proposedClasses[allocation[i]][0] == pref:
                    satisfied += 1
                    break
                count += 1
    return allocation if satisfied >= minimumSatisfaction else None



"""
-----------------------------------------------------------------------------------------------
Question 2
-----------------------------------------------------------------------------------------------
"""
import math

class Bad_AI:
    """
    Class Description:
    This is the main interface class. It models an AI that stores a list of valid words and is capable of checking
    if a given suspicious word is exactly one character.

    Approach Description:
    The Bad_AI class builds a trie using the input list of words. It provides a method check_word() that determines whether a suspicious word matches any word in the trie with exactly one character substituted.
    The problem is modeled using a trie, which allows fast lookups. A custom recursive search function in the trie walks through each position, tracking whether a substitution has been made. 
    Only if exactly one substitution occurs and the word exists in the trie is it added to the result. This ensures all matched words differ by exactly one letter.

    Input:
       - list_words: A list of valid words to be stored and checked against.

    Output/Postcondition: Initializes the Trie with all valid words for future querying.

    Time Complexity: 
    - __init__: O(C), where C is the total number of characters in list_words.

    Time Complexity Analysis:
       - Each character of each word is inserted exactly once in the Trie.
       - Since we are storing a total of C characters, the cost is linear in C.

    Space Complexity: O(C), to store all characters in the Trie.

    Space Complexity Analysis: 
    Each node represents a character; the total number of nodes is proportional to the number of characters.
    """
    def __init__(self, list_words):
        self.trie = WordTrie()
        for word in list_words:
            self.trie.insert(word)
    
    def check_word(self, sus_word):
        """
        Function Description: Returns a list of words that have exactly one character different from the input suspicious word.
        
        Approach Description:
        Only substitutions (no insertions/deletions) are allowed. 
        Traverse the trie depth-first, tracking whether a single character has been replaced. Upon reaching the word's end with 
        exactly one substitution, add the reconstructed word to the results.

        Input:
        - sus_word(str): The suspicious word to be compared.

        Output: 
        - List[str]: List of words in the Trie that differ from sus_word by exactly one substitution.

        Time Complexity: O(J*N) + O(X), where 
        - J = length of sus_word
        - N = number of words
        - X = total characters in all matching results

        Time Complexity Analysis:
        - Traverses Trie paths matching word length J.
        - Only explores words of the same length, avoiding unnecessary branches.

        Space Complexity: O(X)

        Space Complexity Analysis: Stores X characters (all matched words) in the result list.
        """
        return self.trie.match_with_one_substitution(sus_word)
    
    

class TrieNode:
    """
    A node in the Trie representing one character.

    Approach Description:
       - Models a single character position in a word.
       - Children are stored in a fixed-size array for fast index-based access.
       - Designed to support character-by-character insertions and lookups.
       - A boolean flag indicates if a complete word ends at this node.

    Output/Postcondition:
       - Creates a new node with 26 None children and is_end=False.

    Time Complexity: O(1)

    Time Complexity Analysis: Initialization involves constant-time operations only.

    Space Complexity: O(1)

    Space Complexity Analysis:
    Each node allocates space for 26 pointers regardless of usage.
    """
    def __init__(self):
        self.children = [None] * 26 
        self.is_end = False


class WordTrie:
    """
    Class Description:
    Implements a Trie to store a list of words and support searching
    for words that have exactly one substitution difference from a given word.

    Approach Description:
    The trie allows efficient word insertion and flexible recursive search.
    The insert() method builds the trie from the input words.
    The match_with_one_substitution() method initializes a result list and calls _search(), 
    a recursive function that:
    - Traverses the trie character-by-character.
    - Tracks whether a character was substituted.
    - On reaching the end of the word with exactly one substitution and is_end=True, adds the result.
    A path list is used to build candidate words during traversal efficiently, avoid string concatenation overhead.
   
    Output/Postcondition: 
    - Trie is built with inserted words. 
    - Can return list of words with one substitution from input.

    Time Complexity:
       - insert: O(L), where L is the length of the word being inserted.
       - match_with_one_substitution: O(J*N) + O(X)

    Time Complexity Analysis:
       - insert: Each character added sequentially to Trie, no backtracking.
       - match: For J-character word, explore up to 26 branches at each level, bounded by N.

    Space Complexity:
       - insert: O(C), for total characters stored
       - match: O(X), where X is the total output character count.

    Space Complexity Analysis:
       - Trie grows as O(C) total for all insertions.
       - path array is reused; result list only grows with successful matches.
    """

    def __init__(self):
        self.root = TrieNode()
        self.results = []
    

    def insert(self,word):
        """
        Function description:
        Inserts a lowercase word into the trie by creating necessary child nodes along the path 
        and marking the final node as a word endpoint. 
        
        Approach description:
        Starting from the root, for each character in the word, the method calculates the index 
        corresponding to the character and moves to or creates the child node at that index.
        After all characters are processed, it sets the end-of-word flag at the final node.
        
        Input: word(str): The word to be inserted

        Output/ Postcondition: Trie updated with this word.
        
        Time Complexity: O(L), where L is the length of the word.
         
        Time Complexity Analysis: Each character is processed once, and child node access or creation is
        constant time. 
         
        Space Complexity: O(L)
         
        Space Complexity Analysis: New nodes are created only when the path does not already exist, contributing up to
        one node per character.
        """
        current = self.root 
        for ch in word:
            index = ord(ch) - ord('a')
            if current.children[index] is None: 
                current.children[index] = TrieNode()
            current = current.children[index]
        current.is_end = True 
    
    def match_with_one_substitution(self, sus_word):
        """
        Function Description: 
        Initiates a search for all words in the Trie that differ from 'sus_word' by exactly one substitution.
        
        Approach Description: 
        Recursively traverses the Trie. At each character, either match it or try a substitution (only once). 
        When the word ends, only include it if exactly one substitution was used.
        
        Input: sus_word (str): The suspicious word to compare.
        
        Output: List[str]: All valid words differing by one substitution.
        
        Time Complexity: O(J*N) + O(X), where J is the length of the suspicious word, 
        N is the number of words, and X is the total length of matched words.
        
        Space Complexity: O(X)
        """
        self.results = []
        path = [''] * len(sus_word)
        self._search(self.root, sus_word, 0, 0, path)
        return self.results 
    
    def _search(self,node,word,pos, subs_used, path):
        """
        Function Description:
        Recursive explores the Trie to find matching words with exactly one character substitution.

        Approach Description:
           - At each character position, check all 26 possible children.
           - If character matches, no substitution added.
           - If character differs, increment substitution count.
           - If substitutions > 1, stop search early (prune).
           - On reaching end of word with exactly 1 substitution and valid path, add to result.

        Input:
           - node (TrieNode): Current node in the Trie.
           - word (str): The suspicious word.
           - pos (int): Current character position.
           - subs_used (int): Count of substitutions made so far.
           - path (List[str]): Accumulated characters along the path.

        Output/Postcondition:
           - If a match is found with exactly one substitution, add to results.

        Time Complexity: O(J*N)

        Time Complexity Analysis:
           - Each character in word can result in 26 branches, but only if ≤1 substitution used.
           - Total branches pruned early; full traversal over N valid Trie paths of length J.

        Space Complexity: O(J)

        Space Complexity Analysis: O(J) stack space (path + recursion depth).
        """
        if node is None:
            return 
        if subs_used > 1:
            return 
        
        if pos == len(word):
            if node.is_end and subs_used == 1:
                self.results.append(''.join(path))
            return 
        
        target_index = ord(word[pos]) - ord('a')
        for i in range(26):
            next_node = node.children[i]
            if next_node is None:
                continue 

            # Increase substitution count only if char differs 
            new_subs_used = subs_used + (0 if i == target_index else 1)

            if new_subs_used <= 1:
                path[pos] = chr(i + ord('a'))
                self._search(next_node, word, pos + 1, new_subs_used, path)
