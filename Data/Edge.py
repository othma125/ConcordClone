class Edge:
    ''' An edge is a pair of vertices (x, y) '''
    def __init__(self, x: int, y: int = None):
        ''' If y is None, then it is a fake edge (x == y) '''
        if y is None:
            y = x
        self.X = x
        self.Y = y

    def __hash__(self) -> int:
        ''' Compute a hash code for the edge for using it in a set or as a key in a dictionary '''
        return 31 * self.X + self.Y

    def inverse(self) -> 'Edge':
        ''' Return the inverse of the edge (y, x) '''
        return Edge(self.Y, self.X)

    def clone(self) -> 'Edge':
        ''' Return a clone of the edge '''
        return Edge(self.X, self.Y)

    def __str__(self) -> str:
        ''' Return a string representation of the edge '''
        return f"{self.X} {self.Y}"

    def get_x(self) -> int:
        ''' Return the first vertex of the edge '''
        return self.X

    def get_y(self) -> int:
        ''' Return the second vertex of the edge '''
        return self.Y

    def is_fake(self) -> bool:
        ''' Check if the edge is a fake edge (x == y) '''
        return self.X == self.Y

    def is_equals_to(self, edge: 'Edge') -> bool:
        ''' Check if two edges are equal '''
        return self.X == edge.X and self.Y == edge.Y

    def __eq__(self, other: object) -> bool:
        ''' Check if two edges are equal '''
        if self is other:
            return True
        if other is None or type(self) != type(other):
            return False
        return self.is_equals_to(other)
