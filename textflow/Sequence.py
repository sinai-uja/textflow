import os
from typing import Optional


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
    """Summary of class here.

    Longer class information...
    Longer class information...

    Attributes:
        id: ...
        text: ...
        sequences: ...
    """
    def __init__(self, format: str, item: object, id: Optional[str] = None):
        """Creates a sequence from an input object.

        Args:
            format: A string containing the input data's type.
            item: An object representing the input data. It can be a string for a
            string format or a file path for a text format.
            id: A string to overwrite the default's sequence id.
        """
        VALID_FORMATS = ("string", "text")
        
        if format not in VALID_FORMATS:
            raise ValueError(
                f"{format} is not a valid format. Valid formats: {VALID_FORMATS}"
            )

        # Splits string text by whitespace
        if format == "string":
            if not isinstance(item, str):
                raise ValueError(f"{item} is not an instance of string")
            self.id = id if id else "string"
            self.text = item
            self.sequences = item.split(" ")

        # Splits file text by \n
        if format == "text":
            self.id = id if id else os.path.basename(item).split(".")[0]
            with open(item, "r") as f:
                self.text = f.read()
            self.sequences = self.text.split("\n")

    def __str__(self):
        return self.text

    def __repr__(self):
        values = ", ".join([sequence.__repr__() for sequence in self.sequences])
        return (
            "Sequence(\n"
            f"  id: {self.id}\n"
            f"  sequences: {values}\n"
            ")"
        )

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