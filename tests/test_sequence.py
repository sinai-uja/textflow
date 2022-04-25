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
        , 
        pytest.param(
            Sequence("directory", "tests/data"),
            " "
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
            1
        ), 
        pytest.param(
            Sequence("text", "tests/data/doc_1.txt"),
            1
        ),
        pytest.param(
            Sequence("directory","tests/data" ), 
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
            [Sequence("token",tkn) for tkn in "Lorem ipsum".split(" ")]
        ), 
        pytest.param(
            Sequence("text", "tests/data/doc_1.txt"),
            [Sequence("token",tkn) for tkn in "Lorem ipsum".split(" ")]
        ),
        pytest.param(
            Sequence("directory","tests/data" ), 
            [Sequence().initFromDocument("tests/data/doc_1.txt","tokens","token")]
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
            Sequence("string", "Lorem ipsum dolor sit amet").children["tokens"]
        ), 
        pytest.param(
            Sequence("text", "tests/data/doc_1.txt"),
            Sequence("text", "tests/data/doc_1.txt").children["tokens"]
        ),
        pytest.param(
            Sequence("directory","tests/data" ), 
            Sequence("directory","tests/data" ).children["files"]
        )
    ]
)
def test_getitem(sequence, expected):
    assert sequence[0] == expected

@pytest.mark.parametrize(
    "sequence, expected", 
    [
        pytest.param(
            Sequence("string", "Lorem ipsum dolor sit amet"),
            (1,["tokens"])
        ), 
        pytest.param(
            Sequence("text", "tests/data/doc_1.txt"),
            (1, ["tokens"])
        ),
        pytest.param(
            Sequence("directory","tests/data" ), 
            (2, ["files", "tokens"])
        )
    ]
)
def test_get_depth(sequence, expected):
    assert sequence.depth() == expected


def test_filter():
    pass