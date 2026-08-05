from ai_assignment.core.constants import (
    MIXED_SENTIMENT_CONNECTORS,
    NEUTRAL_KEYWORDS,
    NEUTRAL_SHORT_RESPONSES,
)
from ai_assignment.core.lexicons import (
    get_negative_cue_phrases,
    get_negative_cue_words,
    get_neutral_phrases,
    get_positive_cue_words,
    get_positive_phrases,
)
from ai_assignment.core.preprocessing import normalize_short_response


def is_neutral_short_response(text):
    normalized_text = normalize_short_response(text)
    compact_text = normalized_text.replace(" ", "")
    return (
        normalized_text in NEUTRAL_SHORT_RESPONSES
        or compact_text in NEUTRAL_SHORT_RESPONSES
    )


def is_neutral_review(text, cleaned_text):
    normalized_text = normalize_short_response(text)
    cleaned_words = set(cleaned_text.split())

    if is_neutral_short_response(text):
        return True

    if cleaned_words.intersection(NEUTRAL_KEYWORDS):
        return True

    if has_mixed_sentiment(text, cleaned_text):
        return True

    return any(phrase in normalized_text for phrase in get_neutral_phrases())


def has_mixed_sentiment(text, cleaned_text):
    normalized_text = normalize_short_response(text)
    normalized_words = set(normalized_text.split())
    cleaned_words = set(cleaned_text.split())
    has_connector = bool(
        normalized_words.intersection(MIXED_SENTIMENT_CONNECTORS)
        or cleaned_words.intersection(MIXED_SENTIMENT_CONNECTORS)
    )

    if not has_connector:
        return False

    has_positive_cue = bool(cleaned_words.intersection(get_positive_cue_words()))
    has_negative_cue = (
        bool(cleaned_words.intersection(get_negative_cue_words()))
        or any(phrase in normalized_text for phrase in get_negative_cue_phrases())
    )

    return has_positive_cue and has_negative_cue


def is_positive_review(text):
    normalized_text = normalize_short_response(text)
    return any(phrase in normalized_text for phrase in get_positive_phrases())
