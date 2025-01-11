import sys
from utils import display_puzzle
from typing import List, Callable, Dict,Union, Set, Any

def get_and_filter_line(
		puzzle: List[int],
		size :int,
		modified_line_index: List[int],
		column_indices: List[List[int]],
		goal_lines : Dict[int, Dict[str, Union[List[int], Set[int]]]],
		is_row=True) -> List[List[int]]:
	filtered_lines = []
	"""
    Extracts and filters rows or columns from the puzzle based on their corresponding goal lines.

    :param puzzle: Flattened list representing the puzzle's current state.
    :param size: Size of the puzzle (e.g., 4 for a 4x4 puzzle).
    :param modified_line_index: List of indices of the modified rows or columns.
    :param column_indices: Precomputed list of indices for each column in the puzzle.
    :param goal_lines: Goal state information for rows or columns.
    :param is_row: Flag to indicate whether to process rows (True) or columns (False).
    :return: List of filtered rows or columns.
    """
	# Iterate over the modified line indices
	for index in modified_line_index:
		if is_row:
			# Extract the row
			line = puzzle[index * size:(index + 1) * size]
		else:
			# Extract the column
			line = [puzzle[k] for k in column_indices[index]]
		# Filter the line based on goal tiles
		filtered_line = [tile for tile in line if tile != 0 and tile in goal_lines[index]['set']]
		filtered_lines.append(filtered_line)
	return filtered_lines

def linear_conflict_on_multiple_lines(
		filtered_lines: List[List[int]], 
		goal_lines: List[List[int]],
		goal_indices: Dict[int, int]
	) -> int:
	"""
	Computes the total linear conflict for multiple rows or columns.

    :param filtered_lines: List of filtered rows/columns list from the current state.
    :param goal_lines: List of corresponding goal rows/columns list.
    :param goal_indices: Dictionary of goal indices for each tile.
    :return: Total linear conflict distance across all rows/columns.
	"""
	total_conflict = 0

	for filtered_line, goal_line in zip(filtered_lines, goal_lines):
		conflict_count = 0
		size_current_line = len(filtered_line)

		for index in range(size_current_line):
			tile_current = filtered_line[index]
			if tile_current == goal_line[index]:
				continue

			goal_position_current = goal_indices[tile_current]

			for j in range(index + 1, size_current_line):
				other_tile = filtered_line[j]
				goal_position_other = goal_indices[other_tile]

				if (goal_position_current < goal_position_other and index > j) or (goal_position_current > goal_position_other and index < j):
					conflict_count += 1

		total_conflict += 2 * conflict_count
	return total_conflict