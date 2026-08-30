from ai_assignment.core.constants import (
    MIXED_SENTIMENT_CONNECTORS,
    NEUTRAL_KEYWORDS,
    NEUTRAL_SHORT_RESPONSES,
    POSITIVE_SHORT_RESPONSES,
    STRONG_NEGATIVE_KEYWORDS,
)
from ai_assignment.core.lexicons import (
    get_negative_cue_phrases,
    get_negative_cue_words,
    get_neutral_phrases,
    get_positive_cue_words,
    get_positive_phrases,
)
from ai_assignment.core.preprocessing import normalize_short_response


NEGATION_WORDS = {"not", "no", "never"}
NEGATION_SCOPE_LIMIT = 3


def is_neutral_short_response(text):
    normalized_text = normalize_short_response(text)
    compact_text = normalized_text.replace(" ", "")
    return (
        normalized_text in NEUTRAL_SHORT_RESPONSES
        or compact_text in NEUTRAL_SHORT_RESPONSES
    )


def is_positive_short_response(text):
    normalized_text = normalize_short_response(text)
    compact_text = normalized_text.replace(" ", "") #remove space between words to check if the compacted text is in the positive short responses set
    return (
        normalized_text in POSITIVE_SHORT_RESPONSES
        or compact_text in POSITIVE_SHORT_RESPONSES
    )


def has_positive_keyword(text, cleaned_text):
    normalized_words = set(normalize_short_response(text).split())
    cleaned_words = set(cleaned_text.split())

    if normalized_words.intersection(NEGATION_WORDS):
        return False

    if cleaned_words.intersection(STRONG_NEGATIVE_KEYWORDS):
        return False

    return has_positive_cue(cleaned_words)


def has_positive_cue(words):
    positive_words = get_positive_cue_words().union(POSITIVE_SHORT_RESPONSES)
    return bool(set(words).intersection(positive_words))


def has_negative_cue(words):
    negative_words = get_negative_cue_words().union(STRONG_NEGATIVE_KEYWORDS)
    return bool(set(words).intersection(negative_words))


def has_negated_positive_cue(cleaned_text):
    words = cleaned_text.split()
    positive_words = get_positive_cue_words().union(POSITIVE_SHORT_RESPONSES)

    for index, word in enumerate(words):
        if word not in NEGATION_WORDS:
            continue

        nearby_words = words[index + 1 : index + 1 + NEGATION_SCOPE_LIMIT]
        if set(nearby_words).intersection(positive_words):
            return True

    return False


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

    if any(phrase in normalized_text for phrase in get_negative_cue_phrases()):
        return has_positive_cue(cleaned_words)

    return has_connector_followed_by_negation(cleaned_text)


def has_connector_followed_by_negation(cleaned_text):
    words = cleaned_text.split()

    for index, word in enumerate(words):
        if word not in MIXED_SENTIMENT_CONNECTORS:
            continue

        words_before_connector = words[:index]
        words_after_connector = words[index + 1 :]

        before_is_positive = has_positive_cue(words_before_connector)
        after_is_positive = has_positive_cue(words_after_connector)
        before_is_negative = has_negative_clause(words_before_connector)
        after_is_negative = has_negative_clause(words_after_connector)

        if before_is_positive and after_is_negative:
            return True

        if before_is_negative and after_is_positive:
            return True

    return False


def has_negative_clause(words):
    clause_text = " ".join(words)
    if any(phrase in clause_text for phrase in get_positive_phrases()):
        return False

    if has_negative_cue(words):
        return True

    if has_negated_positive_cue(clause_text):
        return True

    return bool(set(words).intersection(NEGATION_WORDS))


def is_positive_review(text, cleaned_text):
    normalized_text = normalize_short_response(text)
    if any(phrase in normalized_text for phrase in get_positive_phrases()):
        return True

    return is_positive_short_response(text) or has_positive_keyword(
        text,
        cleaned_text,
    )


def is_negative_review(text, cleaned_text):
    normalized_text = normalize_short_response(text)
    cleaned_words = set(cleaned_text.split())

    if any(phrase in normalized_text for phrase in get_negative_cue_phrases()):
        return True

    if has_negated_positive_cue(cleaned_text):
        return True

    return bool(cleaned_words.intersection(STRONG_NEGATIVE_KEYWORDS))
