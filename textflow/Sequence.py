import os
from typing import Optional


class SequenceIterator:
    def __init__(self, children, sequences):
        self.idx = 0
        self.children = children
        self.sequences = sequences
    
    def __iter__(self):
        return self

    def __next__(self):
        self.idx += 1
        try:
            #return self.data[self.idx-1]
            return {
                "child": self.children[self.idx-1],
                "sequence": self.sequences[self.idx-1]
            }
        except IndexError:
            self.idx = 0
            raise StopIteration


_VALID_FORMATS = ["string", "text", "token", None]

class Sequence:
    """Summary of class here.

    Longer class information...
    Longer class information...

    Attributes:
        id: ...
        text: ...
        sequences: ...
    """
    def __init__(self, format: Optional[str] = None, src: Optional[object] = None, id: Optional[str] = None):
        """Creates a sequence from an input object.

        Args:
            format: A string containing the input data's type.
            src: An object representing the input data. It can be a string for a
            string format or a file path for a text format.
            id: A string to overwrite the default's sequence id.
        """
        
        if format not in _VALID_FORMATS:
            raise ValueError(
                f"{format} is not a valid format. Valid formats: {_VALID_FORMATS}"
            )

        if format == "token":
            raise ValueError(
                f"Tokens can not be split"
            )

        # Empty sequence
        if format is None:
            self.id = id
            self.text = None
            self.children = []
            self.sequences = []

        # Splits string text by whitespace
        if format == "string":
            if not isinstance(src, str):
                raise ValueError(f"{src} is not an instance of string")
            self.id = id if id else "string"
            self.text = src
            self.children = [("token", token_src) for token_src in src.split(" ")]
            self.sequences = [Sequence() for _ in self.children]

        # Splits file text by \n
        if format == "text":
            self.id = id if id else os.path.basename(src).split(".")[0]
            with open(src, "r") as f:
                self.text = f.read()
            self.children = [("string", line_src) for line_src in self.text.split("\n")]
            self.sequences = [Sequence() for _ in self.children]

    def __str__(self):
        return self.text

    def __repr__(self):
        children = ", ".join([child.__repr__() for child in self.children])
        sequences = ", ".join([sequence.__repr__() for sequence in self.sequences])
        return (
            "Sequence(\n"
            f"  id: {self.id}\n"
            f"  text: {self.text}\n"
            f"  children: {children}\n"
            f"  sequences: {sequences}\n"
            ")"
        )

    def __len__(self):
        return len(self.children)

    def __iter__(self):
        return SequenceIterator(self.children, self.sequences)
    
    def __getitem__(self, idx):
        if isinstance(idx, str):  # Get src by string (e.g. seq["doc1"])
            if self.sequences[0] is None:
                    raise ValueError(f"Sequence id '{idx}' not found in {self.sequences}")
            for cont, sequence in enumerate(self.sequences):
                if sequence.id == idx: return {
                    "child": self.children[cont],
                    "sequence": self.sequences[cont]
                }
            raise ValueError(f"Sequence id '{idx}' not found in {self}")

        elif isinstance(idx, int):  # Get src by int (e.g. seq[0])
            if abs(idx) >= len(self.children):
                raise IndexError(f"Sequence index '{idx}' out of range")

            if idx < 0:
                idx = len(self.children) + idx
            
            return {
                "child": self.children[idx],
                "sequence": self.sequences[idx]
            }
        else:   # TODO: Should it support slices (e.g. [2:4])?
            raise TypeError(
                f"Sequence indices must be integers or strings, not {type(idx).__name__}"
            )

    def set_sequence(self, new_sequence):
        print("Setting value...")
        self.id = new_sequence.id
        self.text = new_sequence.text
        self.children = new_sequence.children
        self.sequences = new_sequence.sequences

    def get_depth(self):
        pass    # TODO

    def filter(self, level, criteria):
        pass    # TODO