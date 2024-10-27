# Implémentation de l'algorithme A* et des heuristiques
import heapq
from heuristics import manhattan_distance

def a_star(puzzle, goal, size, heuristic_func):
    """
    Implémente l'algorithme A* pour résoudre le N-puzzle.
    - puzzle : état initial du puzzle sous forme de liste
    - goal : état final (objectif) du puzzle
    - size : taille du puzzle (par exemple, 3 pour 3x3)
    - heuristic_func : fonction heuristique pour estimer la distance au but
    """
    # Initialisation de la file de priorité et des structures de suivi
    open_set = []
    heapq.heappush(open_set, (0, puzzle, [], 0))  # (priorité, état, chemin, g(x))
    visited = set()
    visited.add(tuple(puzzle))
    
    # Statistiques
    states_explored = 0
    max_states_in_memory = 1

    while open_set:
        f, current, path, g = heapq.heappop(open_set)
        states_explored += 1

        # Si nous avons atteint l'état final
        if current == goal:
            return {
                "path": path + [current],
                "moves": len(path),
                "states_explored": states_explored,
                "max_states_in_memory": max_states_in_memory
            }

        # Générer les voisins (états atteignables en un mouvement)
        neighbors = get_neighbors(current, size)
        for neighbor, move in neighbors:
            if tuple(neighbor) not in visited:
                visited.add(tuple(neighbor))
                new_path = path + [current]
                h = heuristic_func(neighbor, goal, size)
                heapq.heappush(open_set, (g + 1 + h, neighbor, new_path, g + 1))

        # Mise à jour de la mémoire maximale utilisée
        max_states_in_memory = max(max_states_in_memory, len(open_set))

    # Si aucun chemin n'a été trouvé
    return None

def get_neighbors(puzzle, size):
    """
    Génère les voisins d'un état de puzzle en échangeant la case vide (0) avec les cases adjacentes.
    - puzzle : état actuel du puzzle sous forme de liste
    - size : taille du puzzle (ex: 3 pour un puzzle 3x3)
    """
    neighbors = []
    zero_index = puzzle.index(0)
    x, y = zero_index % size, zero_index // size

    def swap_and_create(new_x, new_y):
        new_index = new_y * size + new_x
        new_puzzle = puzzle[:]
        new_puzzle[zero_index], new_puzzle[new_index] = new_puzzle[new_index], new_puzzle[zero_index]
        return new_puzzle

    if x > 0:  # Gauche
        neighbors.append((swap_and_create(x - 1, y), "left"))
    if x < size - 1:  # Droite
        neighbors.append((swap_and_create(x + 1, y), "right"))
    if y > 0:  # Haut
        neighbors.append((swap_and_create(x, y - 1), "up"))
    if y < size - 1:  # Bas
        neighbors.append((swap_and_create(x, y + 1), "down"))

    return neighbors

def solve_puzzle(puzzle, size, heuristic):
    """
    Fonction principale pour résoudre le puzzle.
    - puzzle : état initial du puzzle sous forme de liste
    - size : taille du puzzle
    - heuristic : chaîne indiquant l'heuristique à utiliser
    """
    # Définition de l'état final pour un puzzle de taille `size`
    goal = list(range(1, size * size)) + [0]

    # Sélection de la fonction heuristique
    if heuristic == "manhattan":
        heuristic_func = manhattan_distance
    else:
        raise ValueError("Heuristic not supported")

    # Exécution de l'algorithme A*
    return a_star(puzzle, goal, size, heuristic_func)