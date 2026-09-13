"""Normalization is where fastText's format silently corrupts data."""

from node.core.text import (
    normalize_label,
    normalize_text,
    strip_label_prefix,
    to_fasttext_line,
)


def test_newlines_collapse_to_one_line():
    # The headline footgun: a newline inside a document splits it into two
    # training examples, the second one unlabeled, and fastText says nothing.
    text = "first part\nsecond part\r\nthird\tfourth"
    result = normalize_text(text)

    assert "\n" not in result
    assert "\r" not in result
    assert "\t" not in result
    assert result == "first part second part third fourth"


def test_training_line_stays_single_line_for_multiline_document():
    document = "line one\n\nline two\nline three"
    line = to_fasttext_line("3", normalize_text(document))

    assert line.count("\n") == 0
    assert line.startswith("__label__3 ")


def test_punctuation_is_separated():
    # fastText splits on whitespace only, so "word." and "word" would otherwise
    # be unrelated tokens.
    assert normalize_text("Hello, world. Is it?") == "hello , world . is it ?"
    assert normalize_text("a word.").split() == ["a", "word", "."]


def test_hyphens_and_underscores_survive():
    # Deliberate: compounds carry signal and supervised fastText uses no
    # subword n-grams by default, so splitting them only loses information.
    assert normalize_text("state-of-the-art") == "state-of-the-art"
    assert normalize_text("snake_case_name") == "snake_case_name"


def test_label_prefix_inside_text_is_neutralized():
    # A document quoting the training format would otherwise relabel itself.
    result = normalize_text("spam __label__9 more text")
    assert "__label__" not in result

    line = to_fasttext_line("2", result)
    assert line.count("__label__") == 1


def test_label_with_whitespace_cannot_split_the_line():
    # A label containing a space would turn its own tail into document text.
    assert normalize_label("high quality") == "high_quality"
    assert " " not in normalize_label("  spaced  out  ")


def test_lowercased_and_whitespace_collapsed():
    assert normalize_text("  Mixed   CASE   text  ") == "mixed case text"


def test_empty_input_yields_empty_output():
    assert normalize_text("") == ""
    assert normalize_text("   \n\t  ") == ""
    # Punctuation alone is not empty -- it is still a (useless) token.
    assert normalize_text("...") == ". . ."


def test_strip_label_prefix_roundtrip():
    assert strip_label_prefix("__label__7") == "7"
    assert strip_label_prefix("7") == "7"
