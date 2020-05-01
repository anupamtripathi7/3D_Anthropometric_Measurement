import numpy as np


class Node():
    """A node class for A* Pathfinding"""

    def __init__(self, parent=None, idx=None):
        self.parent = parent
        self.idx = idx

        self.g = 0
        self.h = 0
        self.f = 0


def astar(adj_list, start_idx, end_idx, verts):
    # Create start and end node
    start_node = Node(None, start_idx)
    start_node.g = start_node.h = start_node.f = 0
    end_node = Node(None, end_idx)
    end_node.g = end_node.h = end_node.f = 0

    # Initialize both open and closed list
    open_list = [start_node]
    closed_list = []

    # Loop until you find the end
    while open_list:

        # Get the current node
        current_node = open_list[0]
        current_index = 0
        for index, item in enumerate(open_list):
            if item.f < current_node.f:
                current_node = item
                current_index = index

        # Pop current off open list, add to closed list
        open_list.pop(current_index)
        closed_list.append(current_node)

        # Found the goal
        if current_node.idx == end_node.idx:

            path = []
            current = current_node
            while current is not None:
                path.append(current.idx)
                current = current.parent
            return path[::-1] # Return reversed path

        # Add the neighbours nodes to open list
        for neighbour in adj_list[current_node.idx]:
            new_node = Node(current_node, neighbour)
            new_node.g = current_node.g + 1
            new_node.h = np.linalg.norm(verts[new_node.idx] - verts[end_node.idx])
            new_node.f = new_node.g + new_node.h
            open_list.append(new_node)


def main():
    start_idx = 0
    end_idx = 6
    adj_list = {0: [1, 2, 3], 1: [0, 4], 2: [0, 5], 3: [0, 6, 5], 4:[1, 6], 5: [2, 3, 6], 6: [3, 4, 5]}
    verts = np.array([[0, 0, 0], [1, 2, 0],[1, -2, 0], [100, 0, 0], [6, 2, 0], [6, -5, 0], [10, 0, 0]])
    path = astar(adj_list, start_idx, end_idx, verts)

    print(path)


if __name__ == '__main__':
    main()
