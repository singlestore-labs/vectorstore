class IndexList:
    def __init__(self, index_list):
        self.index_list = index_list

    def __getitem__(self, index):
        return self.index_list[index]

    def __len__(self):
        return len(self.index_list)

    def __iter__(self):
        return iter(self.index_list)

    def __repr__(self):
        return f"IndexList({self.index_list})"