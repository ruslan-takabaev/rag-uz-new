"""
Simple rule-based (no NN models involved) tokenization logic
"""

from typing import List

def uzbek_tokenize_word(word: str) -> List[str]:
    """Applies custom tokenization rules to a single Uzbek word."""
    prefixes = ['bad', 'bar', 'ba', 'be', 'bo', 'ham', 'nim', 'no', 'pesh', 'ser']
    noun_suffixes = ['dan', 'da', 'ga', 'ka', 'lar', 'ni', 'ning', 'qa']
    verb_suffixes = ["adi", "ajak", "ay", "aylik", "ayotib", "ayotir", "di", "dir", "gan", "gin", "imiz", "ingiz", "kin", "man", "miz", "moq", "moqchi", "ngiz", "qin", "san", "sin", "siz", "yap", "ylik", "yotib", "yotir"]
    suffixes = ['cha', 'inchi', 'mas', 'mi', 'nchi', 'roq', 'ta']
    if not isinstance(word, str) or ' ' in word: 
        return [word]
    original_word = word
    prefixed_parts, suffixed_parts = [], []
    for prefix in prefixes:
        if word.lower().startswith(prefix):
            prefixed_parts.append(word[:len(prefix)])
            word = word[len(prefix):]
            break
    all_suffixes = sorted(noun_suffixes + verb_suffixes + suffixes, key=len, reverse=True)
    while True:
        found = False
        for suffix in all_suffixes:
            if word.lower().endswith(suffix) and len(word) > len(suffix):
                suffixed_parts.append(word[-len(suffix):])
                word = word[:-len(suffix)]
                found = True
                break
        if not found: 
            break
    return prefixed_parts + [word] + suffixed_parts[::-1] if prefixed_parts or suffixed_parts else [original_word]
