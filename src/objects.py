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


# def get_spatial_relation(b1: list, b2: list) -> str:
#     x1c, y1c = (b1[0] + b1[2]) // 2, (b1[1] + b1[3]) // 2
#     x2c, y2c = (b2[0] + b2[2]) // 2, (b2[1] + b2[3]) // 2
#     dx, dy = x2c - x1c, y2c - y1c

#     relations = []
#     if abs(dy) < 0.2 * (b1[3] - b1[1]):
#         if dx > 0:
#             relations.append("right from")  # справа от
#         else:
#             relations.append("left from")  # слева от
#     if abs(dx) < 0.2 * (b1[2] - b1[0]):
#         if dy > 0:
#             relations.append("below")  # ниже
#         else:
#             relations.append("above")  # выше

#     return ", ".join(relations)


# def get_dominant_color(image, bbox):
#     x1, y1, x2, y2 = bbox
#     crop = image[y1:y2, x1:x2]
#     # Средний цвет в пространстве BGR → переводим в текстовую метку
#     avg_bgr = np.mean(crop.reshape(-1, 3), axis=0)
#     # Конвертация в HSV для более точного определения «цвета»
#     hsv = cv2.cvtColor(np.uint8([[avg_bgr]]), cv2.COLOR_BGR2HSV)[0][0]
#     # Простая дискретизация оттенков по H
#     h = hsv[0]
#     print(h)
#     # ... реализация маппинга h → «red», «green», «blue» и т.д.
#     # TODO:
#     color_name = "red"
#     return color_name
