from solver import precompute_goal_indices, precompute_goal_positions


class PuzzleState:
	def __init__(self, puzzle, goal, size):
		self.puzzle = puzzle
		self.size = size
		self.goal = goal
		self.goal_positions = precompute_goal_positions(self.goal, self.size)
		self.column_indices = [[k * size + j for k in range(size)] for j in range(size)]
		self.row_goal_indices = precompute_goal_indices(self.goal_positions, is_row=True)
		self.col_goal_indices = precompute_goal_indices(self.goal_positions, is_row=False)