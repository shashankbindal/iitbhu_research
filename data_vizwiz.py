"""
VizWiz-VQA loader + text-question filter.

VizWiz (Gurari et al., CVPR 2018) is the real distribution this project targets:
photos taken by blind users, each with a spoken question and 10 crowd answers.
~21% of questions are about reading text in the image — that text subset is what
we train and evaluate on.

This module:
  1. loads the VizWiz annotation JSON,
  2. filters to text-reading questions (keyword heuristic over the question +
     answer consensus; documented as a starting point, refine with VizWiz's
     'text presence' auxiliary annotations when available),
  3. extracts the consensus answer string — the weak text label used by L_vqa
     (we have the *text*, never a clean image).

The dataset download (images + annotations) is large and fetched separately; this
loader points at a local data dir. Logic is unit-tested on an inline sample so it
needs no download to verify.

Annotation entry shape (VizWiz-VQA v2):
    {"image": "VizWiz_val_00000000.jpg",
     "question": "What does this can say?",
     "answers": [{"answer": "diet coke", "answer_confidence": "yes"}, ...],
     "answerable": 1}
"""

import json
import os
import re
from collections import Counter
from dataclasses import dataclass
from typing import List, Optional


# Question phrasings that signal a text-reading intent. Deliberately
# high-recall; precision is refined downstream by also requiring an answerable,
# non-"unanswerable" consensus answer.
_TEXT_CUES = [
    "what does", "what is written", "what does it say", "read", "say on",
    "says on", "label", "name of", "what kind of", "expir", "expiration",
    "instructions", "directions", "what flavor", "what flavour", "brand",
    "title", "what is this product", "ingredients", "how much", "price",
    "what color is the text", "written", "printed", "what's this say",
]

_UNANSWERABLE = {"unanswerable", "unsuitable", "unsuitable image", "", "nothing"}


@dataclass
class VizWizSample:
    image_path: str
    question: str
    answer: str            # consensus answer (weak text label)
    is_text_question: bool
    answerable: bool


def consensus_answer(answers: List[dict]) -> str:
    """Most-agreed answer across the 10 crowd responses, ignoring unanswerable.
    This is the weak supervision target for L_vqa."""
    cleaned = [a.get("answer", "").strip().lower() for a in answers]
    cleaned = [a for a in cleaned if a and a not in _UNANSWERABLE]
    if not cleaned:
        return ""
    return Counter(cleaned).most_common(1)[0][0]


def is_text_question(question: str) -> bool:
    """Heuristic: does the question ask to read text in the image?"""
    q = question.lower().strip()
    return any(cue in q for cue in _TEXT_CUES)


def load_annotations(ann_path: str, images_dir: str, text_only: bool = True,
                     require_answerable: bool = True) -> List[VizWizSample]:
    """Load and filter a VizWiz-VQA annotation file into VizWizSample list."""
    with open(ann_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return _build_samples(raw, images_dir, text_only, require_answerable)


def _build_samples(raw, images_dir, text_only, require_answerable):
    samples = []
    for e in raw:
        q = e.get("question", "")
        answerable = bool(e.get("answerable", 1))
        if require_answerable and not answerable:
            continue
        is_text = is_text_question(q)
        if text_only and not is_text:
            continue
        ans = consensus_answer(e.get("answers", []))
        if text_only and not ans:        # text question with no usable answer -> skip for L_vqa
            continue
        samples.append(VizWizSample(
            image_path=os.path.join(images_dir, e["image"]),
            question=q,
            answer=ans,
            is_text_question=is_text,
            answerable=answerable,
        ))
    return samples


def split_stats(samples: List[VizWizSample]) -> dict:
    return {
        "n": len(samples),
        "n_text": sum(s.is_text_question for s in samples),
        "n_with_answer": sum(bool(s.answer) for s in samples),
    }


if __name__ == "__main__":
    # Unit-test the filtering/consensus logic on an inline sample — no download.
    sample_raw = [
        {"image": "VizWiz_val_0.jpg", "answerable": 1,
         "question": "What does this medicine bottle say?",
         "answers": [{"answer": "paracetamol"}, {"answer": "paracetamol"},
                     {"answer": "para cetamol"}, {"answer": "unanswerable"}]},
        {"image": "VizWiz_val_1.jpg", "answerable": 1,
         "question": "What color is my shirt?",          # NOT a text question
         "answers": [{"answer": "blue"}, {"answer": "blue"}]},
        {"image": "VizWiz_val_2.jpg", "answerable": 0,    # unanswerable -> dropped
         "question": "What is the expiration date?",
         "answers": [{"answer": "unanswerable"}]},
        {"image": "VizWiz_val_3.jpg", "answerable": 1,
         "question": "Can you read the label on this can?",
         "answers": [{"answer": "diet coke"}, {"answer": "diet coke"}, {"answer": "coke"}]},
    ]
    all_s   = _build_samples(sample_raw, "/imgs", text_only=False, require_answerable=False)
    text_s  = _build_samples(sample_raw, "/imgs", text_only=True,  require_answerable=True)

    print("All samples:", split_stats(all_s))
    print("Text subset:", split_stats(text_s))
    for s in text_s:
        print(f"  text-q: {s.question!r:55s} -> answer {s.answer!r}")

    # assertions
    assert len(text_s) == 2, f"expected 2 text questions, got {len(text_s)}"
    assert text_s[0].answer == "paracetamol"           # consensus, ignores 'unanswerable'
    assert text_s[1].answer == "diet coke"
    assert all(s.is_text_question for s in text_s)
    print("\nVizWiz loader logic verified.")
