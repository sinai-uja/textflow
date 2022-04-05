class SequenceIterator:
    def __init__(self, sequences):
        self.idx = 0
        self.data = sequences
    
    def __iter__(self):
        return self

    def __next__(self):
        self.idx += 1
        try:
            return self.data[self.idx-1]
        except IndexError:
            self.idx = 0
            raise StopIteration


class Sequence:
    def __init__(self, id, sequences):
        self.id = id
        self.sequences = sequences

    def __str__(self):
        return f"id: {self.id}, sequences: {self.sequences}"

    def __repr__(self):
        values = ", ".join([sequence.__repr__() for sequence in self.sequences])
        return f"Sequence({values})"

    def __len__(self):
        return len(self.sequences)

    def __iter__(self):
        return SequenceIterator(self.sequences)
    
    def __getitem__(self, i):
        if isinstance(i, str):
            return "Str indexing is not supported yet"  # TODO
        elif isinstance(i, int):
            if i < 0:
                i = len(self.sequences) + i

            if i >= len(self.sequences):
                raise IndexError("Sequence index out of range")
            else:
                return self.sequences[i]
        else:
            invalid_type = type(i)
            raise TypeError(
                "LockableList indices must be integers or slices, not {}"
                .format(invalid_type.__name__)
            )

    def get_depth(self):
        pass    # TODO

    def filter(self, level, criteria):
        pass    # TODO

# TODO: Move these tests to the ./tests folder
if __name__ == "__main__":
    seq = Sequence(1, ["doc1", 3, Sequence(4, [2])])
    print(seq)
    print(seq[2])
    print(seq["a"])
    for s in seq:
        print(s)
    print(seq[20])