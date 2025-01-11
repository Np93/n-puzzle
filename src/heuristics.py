import cProfile
from typing import List, Callable, Dict,Union, Tuple, Set, Any
manhattan_cache = {}

def manhattan_metric(x1:int, y1:int, x2:int, y2:int) -> int:
    """
    Calculates the Manhattan distance heuristic for two tiles of coordinates (x1, y1) and (x2, y2).
    """
    return abs(x1 - x2) + abs(y1 - y2)

def hamming_metric(x1:int, y1:int, x2:int, y2:int) -> int:
    """
    Calculates the hamming distance heuristic for two tiles of coordinates (x1, y1) and (x2, y2).
    """
    if x1 == x2 and y1 == y2:
        return 0
    else:
        return 1

def manhattan_distance(puzzle: List[int], manhattan_precomputed: Dict[int, Dict[int, int]], size: int) -> int:
    """
    Calculates the Manhattan distance heuristic for an N-puzzle.

    The Manhattan distance is the sum of the minimum number of moves (vertical and horizontal) 
    required for each tile to reach its goal position, excluding the blank tile (0).

    Args:
        puzzle (List[int]): The current state of the puzzle represented as a flat list.
            Each element represents a tile, where 0 denotes the blank tile.
        manhattan_precomputed (Dict[int, Dict[int, int]]): A precomputed nested dictionary where:
            - Outer key: Tile value (int).
            - Inner key: Current position (int).
            - Value: Manhattan distance from the current position to the tile's goal position (int).
        size (int): The size of the puzzle (e.g., 4 for a 4x4 puzzle).

    Returns:
        int: The Manhattan distance for the given puzzle state, representing the 
            total number of moves required for all tiles to reach their goal positions.
    """
    total_distance = 0
    for position, tile in enumerate(puzzle):
        if tile != 0:
            total_distance += manhattan_precomputed[tile][position]
    return total_distance

# not done
#  def misplaced_tiles(puzzle, goal, size):
#     misplaced = sum(1 for i, tile in enumerate(puzzle) if tile != 0 and tile != goal[i])
#     return misplaced

# def dynamic_misplaced_heuristic(puzzle, goal, size):
#     misplaced = misplaced_tiles(puzzle, goal, size)
#     # Pondération dynamique qui augmente l'impact de l'heuristique en fonction du nombre de tuiles mal placées
#     weight = 1 + (misplaced / (size * size))
#     return misplaced * weight 

def hamming_distance(puzzle: List[int], goal: List[int], goal_positions: Dict[int, tuple[int, int]], size : int)->int:
    """
    Calculates the Hamming distance heuristic for an N-puzzle.

    The Hamming distance counts the number of tiles that are not in their goal positions, 
    excluding the blank tile (0).

    Args:
        puzzle (List[int]): The current state of the puzzle represented as a flat list.
            Each element represents a tile, where 0 denotes the blank tile.
        goal (List[int]): The goal state of the puzzle represented as a flat list.
            Each element represents a tile, where 0 denotes the blank tile.
        goal_positions (Dict[int, Tuple[int, int]]): A dictionary mapping each tile
            to its goal position as (x, y) coordinates.
        size (int): The size of the puzzle (e.g., 4 for a 4x4 puzzle).

    Returns:
        int: The Hamming distance for the given puzzle state, representing the 
            number of misplaced tiles.
    """
    distance = 0
    for idx, tile in enumerate(puzzle):
        if tile != 0:
            goal_x, goal_y = goal_positions[tile]
            current_x, current_y = idx % size, idx // size
            if (goal_x, goal_y) != (current_x, current_y):
                distance += 1
    return distance

def linear_conflict_distance(puzzle: List[int], goal_positions: Dict[int, tuple[int, int]], size: int) -> int:
    """
    Calculates the linear conflict heuristic for an N-puzzle.

    Linear conflict is an enhancement of the Manhattan distance heuristic.
    It adds additional penalties for tiles that are in the correct row or column
    but out of order relative to their goal positions.

    Args:
        puzzle (List[int]): The current state of the puzzle represented as a flat list.
            Each element represents a tile, where 0 denotes the empty tile.
        goal_positions (Dict[int, Tuple[int, int]]): A dictionary mapping each tile
            to its goal position as (x, y) coordinates.
        size (int): The size of the puzzle (e.g., 4 for a 4x4 puzzle).

    Returns:
        int: The linear conflict heuristic value for the given puzzle state.
    """
    def get_rows_and_columns(puzzle: List[int], size:int)->Tuple[List[List[int]], List[List[int]]]:
        """
        Extracts rows and columns from the puzzle.

        Args:
            puzzle (List[int]): The flat list representation of the puzzle.
            size (int): The size of the puzzle.

        Returns:
            Tuple[List[List[int]], List[List[int]]]: A tuple containing:
                - Rows: List of rows, each represented as a list of integers.
                - Columns: List of columns, each represented as a list of integers.
        """
        rows = [puzzle[i * size : (i + 1) * size] for i in range(size)]
        columns = [[puzzle[k * size + j] for k in range(size)] for j in range(size)]
        return rows, columns

    def filter_candidates(lines: List[List[int]], goal_positions:Dict[int, Tuple[int, int]], is_row: bool = True):
        """
        Filters tiles that belong in the correct row or column.

        Args:
            lines (List[List[int]]): Rows or columns of the puzzle.
            goal_positions (Dict[int, Tuple[int, int]]): The goal positions of each tile.
            is_row (bool, optional): Whether filtering is for rows or columns. Defaults to True.

        Returns:
            List[List[int]]: Filtered rows or columns containing tiles that belong in the current line.
        """
        filtered_lines = []
        for idx, line in enumerate(lines):
            filtered_line = [
                tile
                for tile in line
                if tile != 0 and (goal_positions[tile][1 if is_row else 0] == idx)
            ]
            filtered_lines.append(filtered_line)
        return filtered_lines

    def count_conflicts(tiles: List[List[int]], goal_positions: Dict[int, Tuple[int, int]], is_row: bool = True) -> int:
        """
        Counts the number of linear conflicts in the given rows or columns.

        Args:
            tiles (List[List[int]]): Filtered rows or columns of tiles.
            goal_positions (Dict[int, Tuple[int, int]]): The goal positions of each tile.
            is_row (bool, optional): Whether counting conflicts in rows or columns. Defaults to True.

        Returns:
            int: The number of linear conflicts in the given rows or columns.
        """
        conflicts = 0
        
        for index, line in enumerate(tiles):
            for i in range(len(line)):
                for j in range(i + 1, len(line)):
                    
                    tile1 = tiles[index][i]
                    tile2 = tiles[index][j]
                    
                    if tile1 == 0 or tile2 == 0:
                        continue
                    if is_row:
                        goal_pos1 = goal_positions[tile1][0]
                        goal_pos2 = goal_positions[tile2][0]
                    else:
                        goal_pos1 = goal_positions[tile1][1]
                        goal_pos2 = goal_positions[tile2][1]
                    
                    if goal_pos1 > goal_pos2:
                        conflicts += 1            
        return conflicts

    puzzle_rows, puzzle_columns = get_rows_and_columns(puzzle, size) 
    filtered_rows = filter_candidates(puzzle_rows, goal_positions, is_row=True) 
    filtered_columns = filter_candidates(puzzle_columns, goal_positions, is_row=False)
    row_conflicts = count_conflicts(filtered_rows, goal_positions, True)
    column_conflicts = count_conflicts(filtered_columns, goal_positions, False)

    return 2 * (row_conflicts + column_conflicts)