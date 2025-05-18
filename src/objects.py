from loguru import logger

import spacy
from collections import defaultdict
from spacy.cli.download import download as spacy_download


logger.info("Downloading language model for the spaCy 'en_core_web_sm'")
spacy_download("en_core_web_sm")
nlp = spacy.load("en_core_web_sm")


def get_objects(text: str) -> list[str]:
    doc = nlp(text)

    # Collect noun chunks and their modifiers
    object_details = defaultdict(list)

    for chunk in doc.noun_chunks:
        head = chunk.root.head
        # Filter out pronouns and unimportant chunks
        if chunk.root.pos_ in ["PRON"] or len(chunk.text.strip()) < 3:
            continue

        # Look for adjectives or descriptors around the noun
        description = []
        for token in chunk.root.lefts:
            if token.pos_ in ["ADJ", "NUM"]:
                description.append(token.text)
        for token in chunk.root.rights:
            if token.pos_ in ["ADJ", "NOUN", "NUM"]:
                description.append(token.text)

        # Include prepositional phrases and appositives
        if head.dep_ in ["prep", "amod", "appos"] and head.head != chunk.root:
            description.append(head.text)

        # Final descriptive phrase
        phrase = " ".join(description + [chunk.text])
        object_details[chunk.text].append(phrase)

    # Post-process to filter and deduplicate
    results = set()
    for phrases in object_details.values():
        for phrase in phrases:
            phrase_clean = phrase.strip().lower()
            if any(c.isalpha() for c in phrase_clean):
                results.add(phrase_clean)

    results = sorted(results)
    logger.info(f"Extracted objects: {results}")

    return results
