"""
Typed entity-level NER scoring, shared by both arms.

Ported from the thesis repository (slm-benchmark, src/evaluation/ner_typed.py),
where typed micro-F1 was verified to equal seqeval span F1 on token-aligned
outputs. The fine-tuned arm emits BIO tags over the input tokens and the API
arm emits JSON keyed by type, so the common unit is the (type, surface) pair,
and micro_prf micro-averages exact multiset matches over the corpus.

Changes from the thesis version: the tag map and the entity-type list are
parameters (the defaults are still CoNLL-2003's), because the NER dataset for
this paper (Few-NERD or Broad Twitter Corpus) is not decided yet, and
`entity_types=None` in parse_api_entities accepts every key. micro_prf also
returns per-sentence counts so the bootstrap can resample them.

Pure Python: no torch, so it can be imported by analysis scripts and tests.
"""

import json
import re
from collections import Counter
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

NER_ID2TAG = {
    0: "O", 1: "B-PER", 2: "I-PER", 3: "B-ORG", 4: "I-ORG",
    5: "B-LOC", 6: "I-LOC", 7: "B-MISC", 8: "I-MISC",
}
ENTITY_TYPES = ("PER", "ORG", "LOC", "MISC")

Entity = Tuple[str, str]  # (type, surface)


def bio_tags(tag_string: str, id2tag: Mapping[int, str] = NER_ID2TAG) -> Optional[List[str]]:
    """Map a generated id string such as '[5, 0, 6]' to BIO tags; None if no ids were generated."""
    ints = re.findall(r"\d+", tag_string)
    if not ints:
        return None
    return [id2tag.get(int(t), "O") for t in ints]


def bio_to_entities(tokens: Sequence[str], tags: Sequence[str]) -> List[Entity]:
    """BIO-decode tags over tokens into (type, surface) pairs.

    A span opens on B-, an I- continues an open span (keeping the B- tag's
    type), anything else closes it, and an I- with no open span is treated
    as O. zip() stops at the shorter sequence.
    """
    entities, current, current_type = [], [], None
    for tok, tag in zip(tokens, tags):
        if tag.startswith("B-"):
            if current:
                entities.append((current_type, " ".join(current)))
            current, current_type = [tok], tag[2:]
        elif tag.startswith("I-") and current:
            current.append(tok)
        else:
            if current:
                entities.append((current_type, " ".join(current)))
            current, current_type = [], None
    if current:
        entities.append((current_type, " ".join(current)))
    return entities


def parse_api_entities(
    response_text: str, entity_types: Optional[Sequence[str]] = ENTITY_TYPES
) -> Optional[List[Entity]]:
    """Parse a JSON response keyed by type into (type, surface) pairs, or None if unparseable."""
    try:
        parsed = json.loads(re.sub(r"```json|```", "", str(response_text)).strip())
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(parsed, dict):
        return None
    types = list(parsed) if entity_types is None else entity_types
    out: List[Entity] = []
    for etype in types:
        values = parsed.get(etype, []) or []
        if isinstance(values, str):
            values = [values]
        if not isinstance(values, list):
            return None
        out.extend((etype, str(e)) for e in values)
    return out


def _key(entity: Entity, typed: bool, case_sensitive: bool) -> Tuple:
    etype, surface = entity
    surface = " ".join(surface.split())
    if not case_sensitive:
        surface = surface.lower()
    return (etype, surface) if typed else (surface,)


def sentence_counts(
    gold: Sequence[Entity], pred: Sequence[Entity], typed: bool = True, case_sensitive: bool = False
) -> Tuple[int, int, int]:
    """Return (tp, fp, fn) for one sentence under multiset matching."""
    gc = Counter(_key(tuple(e), typed, case_sensitive) for e in gold)
    pc = Counter(_key(tuple(e), typed, case_sensitive) for e in pred)
    hit = sum((gc & pc).values())
    return hit, sum(pc.values()) - hit, sum(gc.values()) - hit


def prf_from_counts(tp: int, fp: int, fn: int) -> Dict[str, float]:
    """Precision, recall and F1 from summed counts (0.0 where undefined)."""
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def micro_prf(
    gold: Iterable[Sequence[Entity]],
    pred: Iterable[Sequence[Entity]],
    typed: bool = True,
    case_sensitive: bool = False,
) -> Dict[str, object]:
    """Corpus-level micro precision/recall/F1 over per-sentence entity lists (multiset matching)."""
    counts = [sentence_counts(g, p, typed, case_sensitive) for g, p in zip(gold, pred)]
    tp = sum(c[0] for c in counts)
    fp = sum(c[1] for c in counts)
    fn = sum(c[2] for c in counts)
    return {**prf_from_counts(tp, fp, fn), "tp": tp, "fp": fp, "fn": fn, "per_sentence": counts}


def lenient_sentence_f1(gold: Sequence[Entity], pred: Sequence[Entity]) -> float:
    """The thesis instrument: set F1 over lowercased surfaces, types dropped, 1.0 if both empty."""
    g = {s.lower().strip() for _, s in gold}
    p = {s.lower().strip() for _, s in pred}
    if not g and not p:
        return 1.0
    tp = len(g & p)
    precision = tp / len(p) if p else 0.0
    recall = tp / len(g) if g else 0.0
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0
