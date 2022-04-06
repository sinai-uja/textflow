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
    def __init__(self, object):
        # TODO: Extraer id y sequences a partir del object de cualquier forma que se nos ocurra
        # ver: https://huggingface.co/docs/datasets/v2.0.0/en/package_reference/loading_methods#datasets.load_dataset
        if isinstance(object, str):
            self.id = object
        else:
            self.id = "collection"
        self.sequences = ["subcollection_1", "subcollection_2", "subcollection_3"]

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
            for sequence in self.sequences:
                if isinstance(sequence, Sequence):
                    if sequence.id == i: return sequence
            raise ValueError(f"Sequence index '{i}' not found")
        elif isinstance(i, int):
            if i < 0:
                i = len(self.sequences) + i

            if i >= len(self.sequences):
                raise IndexError(f"Sequence index '{i}' out of range")
            else:
                return self.sequences[i]
        else:   # TODO: Should it support slices (e.g. [2:4])?
            invalid_type = type(i)
            raise TypeError(
                f"Sequence indices must be integers or strings, not {invalid_type.__name__}"
            )

    def get_depth(self):
        pass    # TODO

    def filter(self, level, criteria):
        pass    # TODO