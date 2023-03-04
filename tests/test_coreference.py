import string

import pytest

from relatio.preprocessing import Preprocessor

# import time


def test_CE():
    demo_sentences = [
        "Victoria Chen , CFO of Megabucks Banking, saw her million, "
        "as the 38-year-old pay jump to $2.3 became the company's president. "
        "It is widely known that she came to Megabucks from rival Lotsabucks.",
        "Barack Obama nominated Hillary Rodham Clinton as his secretary of state on Monday. "
        "He chose her because she had foreign affairs experience as a former First Lady.",
    ]
    alphabet_string = string.ascii_lowercase
    alphabet_list = list(alphabet_string) + ["rt"]

    p = Preprocessor(
        spacy_model="en_core_web_sm",
        remove_punctuation=True,
        remove_digits=True,
        lowercase=True,
        lemmatize=True,
        remove_chars=[
            '"',
            "-",
            "^",
            ".",
            "?",
            "!",
            ";",
            "(",
            ")",
            ",",
            ":",
            "'",
            "+",
            "&",
            "|",
            "/",
            "{",
            "}",
            "~",
            "_",
            "`",
            "[",
            "]",
            ">",
            "<",
            "=",
            "*",
            "%",
            "$",
            "@",
            "#",
            "’",
        ],
        stop_words=alphabet_list,
        n_process=-1,
        batch_size=100,
    )
    result = p.coreference_resolution(demo_sentences, progress_bar=False)

    assert type(result) == list
    assert len(result) >= 1


if __name__ == "__main__":
    pytest.main(["-v", __file__])
