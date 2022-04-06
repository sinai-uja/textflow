import pytest
from textflow.Sequence import Sequence

class CustomSequence(Sequence):
    def __init__(self, text: str):
        self.id = "root"
        self.sequences = text.split(" ")

@pytest.fixture
def sequence():
    return CustomSequence("Esto es una prueba")

def test_str(sequence):
    assert str(sequence) == "id: root, sequences: ['Esto', 'es', 'una', 'prueba']"

def test_repr(sequence):
    assert repr(sequence) == "Sequence('Esto', 'es', 'una', 'prueba')"

def test_len(sequence):
    assert len(sequence) == 4

def test_iter(sequence):
    assert list(sequence) == ["Esto", "es", "una", "prueba"]

def test_getitem(sequence):
    assert sequence[0] == "Esto"

def test_get_depth():
    pass

def test_filter():
    pass