"""ONE task specification for BANKING77, shared by fine-tuning, evaluation and the API arm.

Pure Python, no dependencies: the API session can import this module to get byte-identical
text. The text below reproduces the CLASSIFICATION template of src/api/prompts.py
(branch claude/api-harness-827c78) with context task_description=BANKING77_DESCRIPTION and
labels=BANKING77_LABELS formatted as "- label" lines (its `_format_context_value`), and its
zero-shot user turn ("Input: {query}\nLabel:"). The fine-tuned model is trained on exactly
this prompt (system + user) with the label as the assistant turn, and evaluated on it.

Decoding for every arm: greedy / temperature 0, at most MAX_NEW_TOKENS tokens.
Parsing for every arm: parse_label below (case-insensitive, separator-insensitive, exact
match to one of the 77 labels, no fuzzy matching). The API's `_norm_label` is identical.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Sequence

TASK = "banking77"
MAX_NEW_TOKENS = 32
TEMPERATURE = 0.0

BANKING77_DESCRIPTION = "You classify online-banking customer messages by intent."

SYSTEM_TEMPLATE = (
    "{task_description}\n\n"
    "Answer with exactly one label from this list, copied verbatim, and nothing else:\n"
    "{labels}"
)
USER_TEMPLATE = "Input: {query}\nLabel:"

# Class-id order of PolyAI/banking77 (read from the dataset card on 2026-11-21).
# One label is mixed-case ("Refund_not_showing_up"): the normaliser must handle it.
BANKING77_LABELS: List[str] = [
    "activate_my_card",
    "age_limit",
    "apple_pay_or_google_pay",
    "atm_support",
    "automatic_top_up",
    "balance_not_updated_after_bank_transfer",
    "balance_not_updated_after_cheque_or_cash_deposit",
    "beneficiary_not_allowed",
    "cancel_transfer",
    "card_about_to_expire",
    "card_acceptance",
    "card_arrival",
    "card_delivery_estimate",
    "card_linking",
    "card_not_working",
    "card_payment_fee_charged",
    "card_payment_not_recognised",
    "card_payment_wrong_exchange_rate",
    "card_swallowed",
    "cash_withdrawal_charge",
    "cash_withdrawal_not_recognised",
    "change_pin",
    "compromised_card",
    "contactless_not_working",
    "country_support",
    "declined_card_payment",
    "declined_cash_withdrawal",
    "declined_transfer",
    "direct_debit_payment_not_recognised",
    "disposable_card_limits",
    "edit_personal_details",
    "exchange_charge",
    "exchange_rate",
    "exchange_via_app",
    "extra_charge_on_statement",
    "failed_transfer",
    "fiat_currency_support",
    "get_disposable_virtual_card",
    "get_physical_card",
    "getting_spare_card",
    "getting_virtual_card",
    "lost_or_stolen_card",
    "lost_or_stolen_phone",
    "order_physical_card",
    "passcode_forgotten",
    "pending_card_payment",
    "pending_cash_withdrawal",
    "pending_top_up",
    "pending_transfer",
    "pin_blocked",
    "receiving_money",
    "Refund_not_showing_up",
    "request_refund",
    "reverted_card_payment?",
    "supported_cards_and_currencies",
    "terminate_account",
    "top_up_by_bank_transfer_charge",
    "top_up_by_card_charge",
    "top_up_by_cash_or_cheque",
    "top_up_failed",
    "top_up_limits",
    "top_up_reverted",
    "topping_up_by_card",
    "transaction_charged_twice",
    "transfer_fee_charged",
    "transfer_into_account",
    "transfer_not_received_by_recipient",
    "transfer_timing",
    "unable_to_verify_identity",
    "verify_my_identity",
    "verify_source_of_funds",
    "verify_top_up",
    "virtual_card_not_working",
    "visa_or_mastercard",
    "why_verify_identity",
    "wrong_amount_of_cash_received",
    "wrong_exchange_rate_for_cash_withdrawal"
]


def format_labels(labels: Sequence[str]) -> str:
    return "\n".join(f"- {l}" for l in labels)


def system_prompt(labels: Sequence[str] = BANKING77_LABELS) -> str:
    return SYSTEM_TEMPLATE.format(
        task_description=BANKING77_DESCRIPTION, labels=format_labels(labels)
    )


def user_prompt(query: str) -> str:
    return USER_TEMPLATE.format(query=query)


def build_messages(query: str, labels: Sequence[str] = BANKING77_LABELS) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": system_prompt(labels)},
        {"role": "user", "content": user_prompt(query)},
    ]


# ------------------------------------------------------------------ normaliser / parser

def norm_label(s: str) -> str:
    """Same as the API's _norm_label: strip, drop wrapping punctuation, lower-case, '_' separators."""
    return re.sub(r"[\s\-]+", "_", s.strip().strip("`'\".,;:*").strip().lower())


def build_lookup(labels: Sequence[str]) -> Dict[str, str]:
    """normalised form -> canonical label. Raises if two labels collide after normalising."""
    lookup: Dict[str, str] = {}
    for lab in labels:
        key = norm_label(lab)
        if key in lookup and lookup[key] != lab:
            raise ValueError(f"labels {lookup[key]!r} and {lab!r} collide after normalisation")
        lookup[key] = lab
    return lookup


_LOOKUP = build_lookup(BANKING77_LABELS)
_PREFIX = re.compile(r"^(output|label|answer|intent)\s*:\s*", re.IGNORECASE)


def parse_label(text: str, lookup: Optional[Dict[str, str]] = None) -> Optional[str]:
    """First non-empty line -> canonical label, or None (unparseable counts as wrong)."""
    lookup = _LOOKUP if lookup is None else lookup
    lines = [ln for ln in text.strip().splitlines() if ln.strip()]
    if not lines:
        return None
    first = _PREFIX.sub("", lines[0].strip())
    return lookup.get(norm_label(first))
