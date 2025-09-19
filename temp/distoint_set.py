class DisjointSet:
    def __init__(self):
        self.parent = {}

    def __str__(self):
        return str(self.parent)

    def make_set(self, x):
        self.parent[x] = x

    def find_set(self, x):
        if x not in self.parent:
            return None

        if x == self.parent[x]:
            return self.parent[x]
        else:
            return self.find_set(self.parent[x])

    def union(self, x, y):
        root_x = self.find_set(x)
        root_y = self.find_set(y)
        if root_x is None or root_y is None:
            return False

        if root_x != root_y:
            self.parent[root_y] = root_x
        return True


# Example usage
if __name__ == "__main__":
    ds = DisjointSet()
    for i in range(1, 6):
        ds.make_set(i)
    print(ds)
    # Output: {1: 1, 2: 2, 3: 3, 4: 4, 5: 5}

    ds.union(3, 4)  # Union sets w/ 3 & 4
    ds.union(1, 2)  # Union sets w/ 1 & 2

    print(ds)
    # Output: {1: 1, 2: 1, 3: 3, 4: 3, 5: 5}

    print(f"\nfind_set(2) = {ds.find_set(2)}")
    print(f"find_set(4) = {ds.find_set(4)}")

    ds.union(3, 1)  # Union sets w/ 3 & 1
    print(ds)
    # Output: {1: 3, 2: 3, 3: 3, 4: 3, 5: 5}

    for i in range(1, 6):
        print(f"find_set({i}) = {ds.find_set(i)}")
    # Output:
    # find_set(1) = 3
    # find_set(2) = 3
    # find_set(3) = 3
    # find_set(4) = 3
    # find_set(5) = 5
