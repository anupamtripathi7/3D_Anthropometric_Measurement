import numpy as np
class Node():
    """A node class for A* Pathfinding"""

    def __init__(self, parent=None, position=None):
        self.parent = parent
        self.position = position

        self.g = 0
        self.h = 0
        self.f = 0

    def __eq__(self, other):
        return self.position.all() == other.position.all()


def astar(adj_list, start_idx, end_idx, verts):
    # Create start and end node
    start_node = Node(None, verts[start_idx])
    print(start_node.position)
    print(start_node)
    start_node.g = start_node.h = start_node.f = 0
    end_node = Node(None, verts[end_idx])
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
        if current_node == end_node:
            path = []
            current = current_node
            while current is not None:
                path.append(current.position)
                current = current.parent
            return path[::-1] # Return reversed path

        # Generate children
        children = []
        new_positions = [verts[neighbour] for neighbour in adj_list[current_node]]
        for new_position in new_positions:
            # Get node position
            node_position = [new_position[0], new_position[1], new_position[2]]

            # Create new node
            new_node = Node(current_node, node_position)

            # Append
            children.append(new_node)


        # Loop through children
        for child in children:

            # Child is on the closed list
            for closed_child in closed_list:
                if child == closed_child:
                    continue

            # Create the f, g, and h values
            child.g = current_node.g + 1
            child.h = ((child.position[0] - end_node.position[0]) ** 2) \
                      + ((child.position[1] - end_node.position[1]) ** 2) \
                      + ((child.position[0] - end_node.position[0]) ** 2)

            child.f = child.g + child.h

            # Child is already in the open list
            for open_node in open_list:
                if child == open_node and child.g > open_node.g:
                    continue

            # Add the child to the open list
            open_list.append(child)


def main():
    start = 1
    end = 6
    adj_list = {0:[1,2], 1:[0,3], 2:[0,4], 3:[1,6], 4:[2,5,6], 5:[0,4,6], 6:[3,4,5]}
    verts = np.array([[0,0,0],[1,2,0],[-1,2,0],[3,0,0],[6,2,0],[6,-5,0],[10,0,0]])
    path = astar(adj_list, 2, 3, verts)

    print(path)

if __name__ == '__main__':
    main()