import json
from functools import lru_cache

from nltk.corpus import opinion_lexicon
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.stem import WordNetLemmatizer

from ai_assignment.core.constants import SENTIMENT_PHRASES_PATH
from ai_assignment.core.preprocessing import ensure_nltk_data


@lru_cache(maxsize=1)
def get_sentiment_cue_words():
    ensure_nltk_data()
    lemmatizer = WordNetLemmatizer()
    positive_words = normalize_lexicon_words(opinion_lexicon.positive(), lemmatizer)
    negative_words = normalize_lexicon_words(opinion_lexicon.negative(), lemmatizer)

    vader_positive_words, vader_negative_words = get_vader_cue_words(lemmatizer)
    positive_words.update(vader_positive_words)
    negative_words.update(vader_negative_words)

    phrase_rules = get_sentiment_phrases()
    positive_words.update(
        normalize_lexicon_words(
            phrase_rules.get("positive_cue_word_overrides", []),
            lemmatizer,
        )
    )
    negative_words.update(
        normalize_lexicon_words(
            phrase_rules.get("negative_cue_word_overrides", []),
            lemmatizer,
        )
    )

    return positive_words, negative_words


def get_vader_cue_words(lemmatizer):
    analyzer = SentimentIntensityAnalyzer()
    positive_words = []
    negative_words = []

    for word, score in analyzer.lexicon.items():
        if score > 0:
            positive_words.append(word)
        elif score < 0:
            negative_words.append(word)

    return (
        normalize_lexicon_words(positive_words, lemmatizer),
        normalize_lexicon_words(negative_words, lemmatizer),
    )


def normalize_lexicon_words(words, lemmatizer):
    return {
        lemmatizer.lemmatize(str(word).lower())
        for word in words
        if str(word).isalpha()
    }


def get_positive_cue_words():
    return get_sentiment_cue_words()[0]


def get_negative_cue_words():
    return get_sentiment_cue_words()[1]


@lru_cache(maxsize=1)
def get_sentiment_phrases():
    with SENTIMENT_PHRASES_PATH.open("r", encoding="utf-8") as phrase_file:
        phrase_rules = json.load(phrase_file)

    return {
        rule_name: set(phrases)
        for rule_name, phrases in phrase_rules.items()
    }


def get_positive_phrases():
    return get_sentiment_phrases()["positive_phrases"]


def get_negative_cue_phrases():
    return get_sentiment_phrases()["negative_cue_phrases"]


def get_neutral_phrases():
    return get_sentiment_phrases()["neutral_phrases"]
