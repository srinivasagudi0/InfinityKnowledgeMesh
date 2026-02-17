"""Entity extraction module using spaCy NLP."""

from __future__ import annotations

import logging
import re
from functools import lru_cache
from typing import Dict, List, Tuple

import spacy
from spacy.cli import download as download_model
from spacy.language import Language

logger = logging.getLogger(__name__)

MODEL_NAME = "en_core_web_sm"
_CAPITALIZED_PATTERN = re.compile(r"\b([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)\b")


def extract_entities(text: str) -> List[Dict[str, str]]:
    """
    Extract named entities from text using spaCy or a simple fallback heuristic.
    
    Args:
        text: The text to extract entities from.
    
    Returns:
        A list of dictionaries, each containing:
            - text: The entity text.
            - label: The entity type/label.
    """
    if not text:
        logger.warning("Empty text provided for entity extraction")
        return []

    nlp, used_fallback = _get_language_pipeline()
    doc = nlp(text)
    entities = [
        {
            "text": ent.text.strip(),
            "label": ent.label_ or "MISC",
        }
        for ent in getattr(doc, "ents", [])
        if ent.text.strip()
    ]

    if not entities and used_fallback:
        entities = _heuristic_entities(text)

    logger.info("Extracted %d entities", len(entities))
    return entities


@lru_cache(maxsize=1)
def _get_language_pipeline() -> Tuple[Language, bool]:
    try:
        logger.debug("Loading spaCy model '%s'", MODEL_NAME)
        return spacy.load(MODEL_NAME), False
    except OSError:
        logger.warning(
            "spaCy model '%s' not found. Attempting automatic download...", MODEL_NAME
        )
        try:
            download_model(MODEL_NAME)
            return spacy.load(MODEL_NAME), False
        except Exception:
            logger.exception(
                "Unable to load '%s'. Falling back to a lightweight heuristic pipeline.",
                MODEL_NAME,
            )
            blank = spacy.blank("en")
            if "sentencizer" not in blank.pipe_names:
                blank.add_pipe("sentencizer")
            return blank, True


def _heuristic_entities(text: str) -> List[Dict[str, str]]:
    matches = []
    seen = set()
    for match in _CAPITALIZED_PATTERN.findall(text):
        cleaned = match.strip()
        if len(cleaned) < 3 or cleaned in seen:
            continue
        seen.add(cleaned)
        matches.append({"text": cleaned, "label": "MISC"})
    return matches
