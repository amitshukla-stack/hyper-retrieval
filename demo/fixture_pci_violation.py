"""
Payment gateway handler — processes card-not-present transactions.
Routes to appropriate acquirer based on card BIN range.
"""
import logging
import requests

logger = logging.getLogger(__name__)


def process_card_payment(card_number, cvv, expiry, amount, merchant_id):
    """Charges a card via the payment gateway."""
    logger.info(f"Processing card payment: card_number={card_number}, amount={amount}")

    # Tokenize before sending downstream
    resp = requests.post("https://vault.internal/tokenize", json={
        "pan": card_number,
        "cvv": cvv,
        "expiry": expiry,
    })
    token = resp.json()["token"]

    logger.debug(f"Card tokenized. cvv={cvv} discarded after tokenization.")  # bug: logs CVV

    return _charge_via_gateway(token, amount, merchant_id)


def _charge_via_gateway(token, amount, merchant_id):
    resp = requests.post("https://gateway.internal/charge", json={
        "token": token,
        "amount": amount,
        "merchant": merchant_id,
    })
    return resp.json()
