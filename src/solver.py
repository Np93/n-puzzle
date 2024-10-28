# Implémentation de l'algorithme A* et des heuristiques
import heapq
from heuristics import manhattan_distance, linear_conflict, dynamic_misplaced_heuristic, hamming_distance
from parser import is_solvable

def combined_heuristic(puzzle, goal, size):
    return manhattan_distance(puzzle, goal, size) + linear_conflict(puzzle, goal, size)

def get_heuristic_function(heuristic_name):
    """
    Renvoie la fonction heuristique en fonction du nom spécifié.
    """
    if heuristic_name == "manhattan":
        return manhattan_distance
    elif heuristic_name == "manhattan_conflict":
        return combined_heuristic
    elif heuristic_name == "dynamic_misplaced":
        return dynamic_misplaced_heuristic
    elif heuristic_name == "hamming":
        return hamming_distance
    else:
        raise ValueError("Heuristic not supported")

def solve_puzzle(puzzle, size, heuristic_name, inversions):
    """
    Sélectionne la meilleure stratégie de résolution en fonction du nombre d'inversions.
    - puzzle : état initial du puzzle sous forme de liste
    - size : taille du puzzle
    - heuristic_name : nom de l'heuristique à utiliser
    - inversions : nombre d'inversions dans le puzzle
    """
    goal = list(range(1, size * size)) + [0]
    
    # Vérification de la solvabilité
    if not is_solvable(puzzle, size):
        print("Le puzzle n'est pas résolvable.")
        return None
    
    # Choix de l'heuristique
    heuristic_func = get_heuristic_function(heuristic_name)
    
    # Choix de la stratégie en fonction des inversions
    if inversions < 50:
        print("Utilisation de A* classique")
        return a_star(puzzle, goal, size, heuristic_func)
    elif 50 <= inversions <= 100:
        print("Utilisation de la Recherche Gloutonne")
        return greedy_search(puzzle, goal, size, heuristic_func)
    else:
        print("Utilisation de IDA* pour de grandes inversions")
        return ida_star(puzzle, goal, size, heuristic_func)

def get_neighbors(puzzle, size, last_move=None):
    neighbors = []
    zero_index = puzzle.index(0)
    x, y = zero_index % size, zero_index // size

    def swap_and_create(new_x, new_y):
        new_index = new_y * size + new_x
        new_puzzle = puzzle[:]
        new_puzzle[zero_index], new_puzzle[new_index] = new_puzzle[new_index], new_puzzle[zero_index]
        return new_puzzle

    if x > 0 and last_move != "right":
        neighbors.append((swap_and_create(x - 1, y), "left"))
    if x < size - 1 and last_move != "left":
        neighbors.append((swap_and_create(x + 1, y), "right"))
    if y > 0 and last_move != "down":
        neighbors.append((swap_and_create(x, y - 1), "up"))
    if y < size - 1 and last_move != "up":
        neighbors.append((swap_and_create(x, y + 1), "down"))

    return neighbors

def a_star(puzzle, goal, size, heuristic_func):
    open_set = []
    heapq.heappush(open_set, (0, puzzle, [], 0, None))  # (priorité, état, chemin, g(x), dernier mouvement)
    visited = set()
    visited.add("".join(map(str, puzzle)))
    states_explored = 0
    max_states_in_memory = 1

    while open_set:
        f, current, path, g, last_move = heapq.heappop(open_set)
        states_explored += 1
        if current == goal:
            return {"path": path + [current], "moves": len(path), "states_explored": states_explored, "max_states_in_memory": max_states_in_memory}
        neighbors = get_neighbors(current, size, last_move)
        for neighbor, move in neighbors:
            neighbor_str = "".join(map(str, neighbor))
            if neighbor_str not in visited:
                visited.add(neighbor_str)
                new_path = path + [current]
                h = heuristic_func(neighbor, goal, size)
                heapq.heappush(open_set, (g + 1 + h, neighbor, new_path, g + 1, move))
        max_states_in_memory = max(max_states_in_memory, len(open_set))
    return None

def greedy_search(puzzle, goal, size, heuristic_func):
    open_set = []
    heapq.heappush(open_set, (heuristic_func(puzzle, goal, size), puzzle, [], None))  # (priorité h(x), état, chemin, dernier mouvement)
    visited = set()
    visited.add("".join(map(str, puzzle)))
    states_explored = 0
    max_states_in_memory = 1

    while open_set:
        h, current, path, last_move = heapq.heappop(open_set)
        states_explored += 1
        if current == goal:
            return {"path": path + [current], "moves": len(path), "states_explored": states_explored, "max_states_in_memory": max_states_in_memory}
        neighbors = get_neighbors(current, size, last_move)
        for neighbor, move in neighbors:
            neighbor_str = "".join(map(str, neighbor))
            if neighbor_str not in visited:
                visited.add(neighbor_str)
                new_path = path + [current]
                heapq.heappush(open_set, (heuristic_func(neighbor, goal, size), neighbor, new_path, move))
        max_states_in_memory = max(max_states_in_memory, len(open_set))
    return None

def ida_star(puzzle, goal, size, heuristic_func):
    threshold = heuristic_func(puzzle, goal, size)
    while True:
        result = dfs(puzzle, goal, size, heuristic_func, 0, threshold, [])
        if isinstance(result, dict):
            return result
        if result == float('inf'):
            return None
        threshold = result

def dfs(puzzle, goal, size, heuristic_func, g, threshold, path):
    f = g + heuristic_func(puzzle, goal, size)
    if f > threshold:
        return f
    if puzzle == goal:
        return {"path": path + [puzzle], "moves": len(path), "states_explored": len(path), "max_states_in_memory": len(path)}
    min_cost = float('inf')
    for neighbor, move in get_neighbors(puzzle, size):
        if path and neighbor == path[-1]:
            continue
        result = dfs(neighbor, goal, size, heuristic_func, g + 1, threshold, path + [puzzle])
        if isinstance(result, dict):
            return result
        min_cost = min(min_cost, result)
    return min_cost