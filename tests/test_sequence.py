import pytest
from textflow.Sequence import Sequence


def test_sequence_wrong_format():
    with pytest.raises(Exception):
        sequence = Sequence("csv", "Lorem ipsum dolor sit amet")


@pytest.mark.parametrize(
    "sequence, expected", 
    [
        pytest.param(
            Sequence("string", "Lorem ipsum dolor sit amet"),
            "Lorem ipsum dolor sit amet"
        ), 
        pytest.param(
            Sequence("text", "tests/data/doc_1.txt"),
            "Lorem ipsum dolor sit amet\nNam lectus turpis"
        )
    ]
)
def test_str(sequence, expected):
    assert str(sequence) == expected


@pytest.mark.parametrize(
    "sequence, expected", 
    [
        pytest.param(
            Sequence("string", "Lorem ipsum dolor sit amet"),
            (
                "Sequence(\n"
                "  id: string\n"
                "  sequences: 'Lorem', 'ipsum', 'dolor', 'sit', 'amet'\n"
                ")"
            )
        ), 
        pytest.param(
            Sequence("text", "tests/data/doc_1.txt"),
            (
                "Sequence(\n"
                "  id: doc_1\n"
                "  sequences: 'Lorem ipsum dolor sit amet', 'Nam lectus turpis'\n"
                ")"
            )
        )
    ]
)
def test_repr(sequence, expected):
    assert repr(sequence) == expected


@pytest.mark.parametrize(
    "sequence, expected", 
    [
        pytest.param(
            Sequence("string", "Lorem ipsum dolor sit amet"),
            5
        ), 
        pytest.param(
            Sequence("text", "tests/data/doc_1.txt"),
            2
        )
    ]
)
def test_len(sequence, expected):
    assert len(sequence) == expected


@pytest.mark.parametrize(
    "sequence, expected", 
    [
        pytest.param(
            Sequence("string", "Lorem ipsum dolor sit amet"),
            ["Lorem", "ipsum", "dolor", "sit", "amet"]
        ), 
        pytest.param(
            Sequence("text", "tests/data/doc_1.txt"),
            ["Lorem ipsum dolor sit amet", "Nam lectus turpis"]
        )
    ]
)
def test_iter(sequence, expected):
    assert list(sequence) == expected


@pytest.mark.parametrize(
    "sequence, expected", 
    [
        pytest.param(
            Sequence("string", "Lorem ipsum dolor sit amet"),
            "Lorem"
        ), 
        pytest.param(
            Sequence("text", "tests/data/doc_1.txt"),
            "Lorem ipsum dolor sit amet"
        )
    ]
)
def test_getitem(sequence, expected):
    assert sequence[0] == expected


def test_get_depth():
    pass


def test_filter():
    pass