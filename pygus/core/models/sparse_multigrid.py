from typing import Any, Sequence
from mesa.space import MultiGrid, GridContent, Coordinate, Agent
from collections import defaultdict

class NestedDefaultDict(defaultdict):
    def __missing__(self, key):
        self[key] = defaultdict(list)
        return self[key]

class SparseMultiGrid(MultiGrid):
    def __init__(self, width: int, height: int, torus: bool) -> None:
        """Create a new grid.

        Args:
            width, height: The width and height of the grid
            torus: Boolean whether the grid wraps or not.
        """
        self.height = height
        self.width = width
        self.torus = torus
        self.num_cells = height * width

        # Internal list-of-lists which holds the grid cells themselves
        self._grid: list[list[GridContent]]
        self._grid = defaultdict(lambda: defaultdict(dict))

        # Flag to check if the empties set has been created. Better than initializing
        # _empties as set() because in this case it would become impossible to discern
        # if the set hasn't still being built or if it has become empty after creation.
        self._empties_built = False

        # Neighborhood Cache
        self._neighborhood_cache: dict[Any, Sequence[Coordinate]] = {}

        # Cutoff used inside self.move_to_empty. The parameters are fitted on Python
        # 3.11 and it was verified that they are roughly the same for 3.10. Refer to
        # the code in PR#1565 to check for their stability when a new release gets out.
        self.cutoff_empties = 7.953 * self.num_cells**0.384

    def place_agent(self, agent: Agent, pos: Coordinate) -> None:
        """Place the agent at the specified location, and set its pos variable."""
        x, y = pos
        if agent.pos is None or agent not in self._grid[x][y]:
            if type(self._grid[x][y]) is not list:
                self._grid[x][y] = []
            self._grid[x][y].append(agent)
            agent.pos = pos
            if self._empties_built:
                self._empties.discard(pos)
                self._empty_mask[agent.pos] = True

    def __getstate__(self):
        state = self.__dict__.copy()
        # Remove unpickleable entries or convert them to pickleable format
        del state['_neighborhood_cache']  # Assuming this is complex or contains unpickleable items
        # Convert defaultdict to a regular dict or another pickleable structure
        state['grid'] = dict((k, dict(v)) for k, v in self._grid.items())
        return state

    def __setstate__(self, state):
        # Restore from state
        self.__dict__.update(state)
        # Reinitialize any removed entries or convert back from pickleable formats
        self._neighborhood_cache = {}  # Reinitialize as needed
        self._grid = defaultdict(lambda: defaultdict(dict), state['grid'])

    @staticmethod
    def grid_default():
        return defaultdict(dict)