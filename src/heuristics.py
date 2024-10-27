# Implémentation des heuristiques admissibles

def manhattan_distance(puzzle, goal, size):
    """
    Calcule la distance de Manhattan entre `puzzle` et `goal`.
    - puzzle : état actuel du puzzle sous forme de liste
    - goal : état final du puzzle sous forme de liste
    - size : taille du puzzle (ex: 3 pour un puzzle 3x3)
    """
    distance = 0
    for i, tile in enumerate(puzzle):
        if tile == 0:
            continue  # Ignore la case vide
        goal_index = goal.index(tile)
        x1, y1 = i % size, i // size
        x2, y2 = goal_index % size, goal_index // size
        distance += abs(x1 - x2) + abs(y1 - y2)
    return distance