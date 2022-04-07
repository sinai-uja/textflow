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


# @pytest.mark.parametrize(
#     "sequence, expected", 
#     [
#         pytest.param(
#             Sequence("string", "Lorem ipsum"),
#             (
#                 "Sequence(\n"
#                 "  id: string\n"
#                 "  text: 'Lorem ipsum'"
#                 "  children: [(string, 'Lorem'), (string, 'ipsum')]"
#                 ")"
#             )
#         ), 
#         pytest.param(
#             Sequence("text", "tests/data/doc_1.txt"),
#             (
#                 "Sequence(\n"
#                 "  id: doc_1\n"
#                 "  sequences: 'Lorem ipsum dolor sit amet', 'Nam lectus turpis'\n"
#                 ")"
#             )
#         )
#     ]
# )
# def test_repr(sequence, expected):
#     assert repr(sequence) == expected


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
            Sequence("string", "Lorem ipsum"),
            {
                "child": ("token", "Lorem"),
                "sequence": Sequence()
            }
        ), 
        pytest.param(
            Sequence("text", "tests/data/doc_1.txt"),
            {
                "child": [("string", "Lorem ipsum dolor sit amet"), ("string", "Nam lectus turpis")],
                "sequence": [Sequence() for _ in range(2)]
            }
        )
    ]
)
def test_iter(sequence, expected):
    assert iter(sequence).__next__() == expected

@pytest.mark.parametrize(
    "sequence, expected", 
    [
        pytest.param(
            Sequence("string", "Lorem ipsum dolor sit amet"),
            {
                "child": ("token", "Lorem"),
                "sequence": Sequence()
            }
        ), 
        pytest.param(
            Sequence("text", "tests/data/doc_1.txt"),
            {
                "chile": ("string", "Lorem ipsum dolor sit amet"),
                "sequence": Sequence()
            }
        )
    ]
)
def test_getitem(sequence, expected):
    assert sequence[0] == expected


def test_get_depth():
    pass


def test_filter():
    pass