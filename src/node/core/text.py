"""Text normalization for fastText's line-oriented training format.

fastText's supervised format is one example per line::

    __label__3 some document text here

That format is fragile in three specific ways, and every function here exists
to defuse one of them:

1. A newline inside a document silently splits it into two examples, the second
   of which has no label. fastText does not complain -- it just trains on
   garbage. So all newlines, carriage returns and tabs become spaces.
2. fastText tokenizes on whitespace *only*. ``word.`` and ``word`` are two
   unrelated tokens, which inflates the vocabulary and smears the signal, so
   punctuation gets spaced out.
3. Any token in the document that starts with ``__label__`` is parsed as a
   label. A document quoting the training format would corrupt its own example,
   so the prefix is neutralized.
"""

from __future__ import annotations

import re

from node.core.types import LABEL_PREFIX

# Deliberately excludes "-" and "_": hyphenated compounds carry signal and
# supervised fastText does not use subword n-grams by default, so splitting
# them would only lose information.
_PUNCTUATION = r"""!"#$%&'()*+,./:;<=>?@[\]^`{|}~"""

_PUNCT_RE = re.compile("([" + re.escape(_PUNCTUATION) + "])")
_WHITESPACE_RE = re.compile(r"\s+")
_LABEL_PREFIX_RE = re.compile(re.escape(LABEL_PREFIX))


def normalize_text(text: str) -> str:
    """Return a single-line, lowercased, whitespace-tokenizable version of `text`.

    Returns an empty string when nothing survives; callers treat that as a
    sample to skip.
    """
    if not text:
        return ""
    # Neutralize before spacing punctuation, so the prefix is still contiguous.
    text = _LABEL_PREFIX_RE.sub(" ", text)
    text = _PUNCT_RE.sub(r" \1 ", text)
    text = _WHITESPACE_RE.sub(" ", text)
    return text.strip().lower()


def normalize_label(label: str) -> str:
    """Return a label safe to place in a ``__label__`` token.

    A label containing whitespace would split into several tokens and silently
    turn the rest of the label into document text, so whitespace collapses to
    underscores.
    """
    label = _WHITESPACE_RE.sub("_", str(label).strip())
    return _LABEL_PREFIX_RE.sub("", label)


def to_fasttext_line(label: str, text: str) -> str:
    """Build one training line. Both parts are assumed already normalized."""
    return f"{LABEL_PREFIX}{label} {text}"


def strip_label_prefix(raw: str) -> str:
    """Turn fastText's ``__label__3`` back into ``3``."""
    return raw[len(LABEL_PREFIX) :] if raw.startswith(LABEL_PREFIX) else raw
