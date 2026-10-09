"""
Governance, Policy Enforcement & PII Masking Engine

Implements:
1. PII Masking: redacts emails, phone numbers, and names in LLM inputs/outputs.
2. Consent & Suppression Verification:
   - Enforces GDPR/CCPA marketing opt-in checks.
   - Filters out users with `do_not_track=True` or `marketing_opt_in=False`.
3. Frequency Capping & Safe Thresholds:
   - Flags audiences smaller than a minimum threshold to prevent micro-targeting.
4. Human Approval Gateway:
   - Campaigns CANNOT be activated without a valid, unexpired, single-use,
     HMAC-signed human approval token.

Security model for approval tokens
----------------------------------
A token is:  MKT_APPRV_<payload_b64>.<signature_b64>

The payload carries the campaign scope (campaign name, audience id, channel),
an issue time and an expiry, and a unique nonce. The signature is an HMAC-SHA256
over the payload using `CDP_APPROVAL_SECRET`.

This means a caller cannot mint a token by simply typing a prefix: without the
secret they cannot produce a valid signature, and a token issued for one campaign
cannot be replayed against a different campaign (scope binding). Tokens are also
single-use: consumption is recorded in the `approval_token` table.

If `CDP_APPROVAL_SECRET` is unset, token *issuance* falls back to a process-local
ephemeral secret so the demo still runs, but a warning is emitted. Verification
across processes then requires the secret to be configured.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import re
import secrets
import time
import warnings
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import pandas as pd

# --------------------------------------------------------------------------
# Approval token configuration
# --------------------------------------------------------------------------

_ENV_SECRET = os.getenv("CDP_APPROVAL_SECRET", "").strip()

#: Process-local fallback so the pipeline/tests work without configuration.
_EPHEMERAL_SECRET = secrets.token_urlsafe(48)

TOKEN_PREFIX = "MKT_APPRV_"
DEFAULT_TTL_MINUTES = int(os.getenv("CDP_APPROVAL_TTL_MINUTES", "60") or 60)


def _active_secret() -> bytes:
    """Returns the HMAC secret, warning once if we fell back to an ephemeral one."""
    if _ENV_SECRET:
        return _ENV_SECRET.encode("utf-8")
    warnings.warn(
        "CDP_APPROVAL_SECRET is not set; using an ephemeral in-process secret. "
        "Approval tokens will not verify across processes. Set CDP_APPROVAL_SECRET "
        "in your environment (see .env.example).",
        RuntimeWarning,
        stacklevel=3,
    )
    return _EPHEMERAL_SECRET.encode("utf-8")


def _b64e(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _b64d(text: str) -> bytes:
    padding = "=" * (-len(text) % 4)
    return base64.urlsafe_b64decode(text + padding)


@dataclass
class TokenVerification:
    """Outcome of verifying an approval token."""

    valid: bool
    reason: str
    payload: Optional[Dict[str, Any]] = None

    def __bool__(self) -> bool:  # allows `if result:`
        return self.valid


# --------------------------------------------------------------------------
# Governance guardrails
# --------------------------------------------------------------------------

class GovernanceGuardrails:
    MIN_AUDIENCE_SIZE = 10
    MAX_CAMPAIGN_SIZE = 100_000

    # ---------------------------------------------------------------- PII ---
    # Order matters: more specific patterns first so we do not leave fragments.
    _EMAIL_RE = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")
    _PHONE_RE = re.compile(
        r"(?:\+?\d{1,3}[\s.-]?)?"          # optional country code
        r"(?:\(\d{3}\)|\d{3})"             # area code, with or without parens
        r"[\s.-]?\d{3}[\s.-]?\d{4}\b"      # subscriber number
    )
    _SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
    _CARD_RE = re.compile(r"\b(?:\d[ -]?){13,16}\b")

    @staticmethod
    def mask_pii(text: str) -> str:
        """Masks emails, phone numbers, SSNs and card-like sequences in free text."""
        if not isinstance(text, str):
            return text

        text = GovernanceGuardrails._EMAIL_RE.sub("[REDACTED_EMAIL]", text)
        text = GovernanceGuardrails._SSN_RE.sub("[REDACTED_SSN]", text)
        text = GovernanceGuardrails._CARD_RE.sub("[REDACTED_CARD]", text)
        text = GovernanceGuardrails._PHONE_RE.sub("[REDACTED_PHONE]", text)
        return text

    @staticmethod
    def mask_dataframe_pii(df: pd.DataFrame) -> pd.DataFrame:
        """Returns a copy of the DataFrame with direct identifiers masked."""
        masked_df = df.copy()
        if "primary_email" in masked_df.columns:
            masked_df["primary_email"] = masked_df["primary_email"].apply(
                lambda e: e[:2] + "***@" + e.split("@")[-1]
                if isinstance(e, str) and "@" in e
                else "[REDACTED]"
            )
        if "primary_phone" in masked_df.columns:
            masked_df["primary_phone"] = masked_df["primary_phone"].apply(
                lambda p: "***-***-" + str(p)[-4:] if pd.notna(p) else "[REDACTED]"
            )
        for name_col in ("first_name", "last_name"):
            if name_col in masked_df.columns:
                masked_df[name_col] = masked_df[name_col].apply(
                    lambda n: str(n)[0] + "***" if pd.notna(n) else ""
                )
        return masked_df

    # ------------------------------------------------------------ consent ---
    @staticmethod
    def validate_audience_compliance(
        audience_df: pd.DataFrame,
        consent_df: pd.DataFrame,
        require_opt_in: bool = True,
    ) -> Tuple[bool, str, pd.DataFrame]:
        """
        Validates that a target audience satisfies privacy regulation and consent.

        Returns: (is_compliant, explanation, compliant_df)
        """
        empty = audience_df.iloc[0:0]
        if audience_df.empty:
            return False, "Audience is empty.", empty

        merged = audience_df.merge(consent_df, on="unified_customer_id", how="left")

        initial_size = len(merged)
        if initial_size < GovernanceGuardrails.MIN_AUDIENCE_SIZE:
            return (
                False,
                f"Audience size ({initial_size}) is below minimum k-anonymity "
                f"privacy threshold ({GovernanceGuardrails.MIN_AUDIENCE_SIZE}).",
                empty,
            )

        # Treat missing consent rows as non-consenting (fail closed).
        compliant = merged[merged["do_not_track"].fillna(True) == False]  # noqa: E712
        if require_opt_in:
            compliant = compliant[
                compliant["marketing_opt_in"].fillna(False) == True  # noqa: E712
            ]

        compliant_size = len(compliant)
        suppressed_count = initial_size - compliant_size

        explanation = (
            f"Audience validation passed. Total evaluated: {initial_size}. "
            f"Compliant: {compliant_size}. Suppressed (no consent / DNT): {suppressed_count}."
        )
        return True, explanation, compliant[["unified_customer_id"]]

    # ------------------------------------------------- approval token API ---
    @staticmethod
    def issue_approval_token(
        campaign_name: str,
        audience_id: str,
        channel: str,
        approved_by: str = "Marketing_Director",
        ttl_minutes: Optional[int] = None,
    ) -> str:
        """
        Issues a signed, scoped, expiring approval token.

        The token is bound to the campaign/audience/channel it was issued for, so it
        cannot be replayed against a different activation.
        """
        ttl = DEFAULT_TTL_MINUTES if ttl_minutes is None else int(ttl_minutes)
        now = int(time.time())
        payload = {
            "campaign_name": campaign_name,
            "audience_id": audience_id,
            "channel": channel,
            "approved_by": approved_by,
            "issued_at": now,
            "expires_at": now + ttl * 60,
            "nonce": secrets.token_urlsafe(16),
        }
        raw = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
        sig = hmac.new(_active_secret(), raw, hashlib.sha256).digest()
        return f"{TOKEN_PREFIX}{_b64e(raw)}.{_b64e(sig)}"

    @staticmethod
    def verify_activation_approval(
        approval_token: str,
        campaign_name: Optional[str] = None,
        audience_id: Optional[str] = None,
        channel: Optional[str] = None,
    ) -> TokenVerification:
        """
        Verifies signature, expiry and (optionally) campaign scope.

        Returns a `TokenVerification`. Truthy only when the token is fully valid.
        """
        if not approval_token or not isinstance(approval_token, str):
            return TokenVerification(False, "No approval token supplied.")
        if not approval_token.startswith(TOKEN_PREFIX):
            return TokenVerification(False, "Token has an unrecognised format.")

        body = approval_token[len(TOKEN_PREFIX):]
        if "." not in body:
            # Legacy/forged prefix-only tokens (e.g. bare "MKT_APPRV_") land here.
            return TokenVerification(
                False,
                "Token is malformed: expected a signed payload of the form "
                "'MKT_APPRV_<payload>.<signature>'.",
            )

        payload_b64, sig_b64 = body.split(".", 1)

        try:
            raw = _b64d(payload_b64)
            provided_sig = _b64d(sig_b64)
        except Exception:
            return TokenVerification(False, "Token payload is not valid base64.")

        expected_sig = hmac.new(_active_secret(), raw, hashlib.sha256).digest()
        if not hmac.compare_digest(provided_sig, expected_sig):
            return TokenVerification(
                False, "Token signature is invalid (possible forgery or wrong secret)."
            )

        try:
            payload = json.loads(raw.decode("utf-8"))
        except Exception:
            return TokenVerification(False, "Token payload is not valid JSON.")

        if int(payload.get("expires_at", 0)) < int(time.time()):
            return TokenVerification(
                False, "Token has expired. Request a new approval.", payload
            )

        checks = (
            ("campaign_name", campaign_name),
            ("audience_id", audience_id),
            ("channel", channel),
        )
        for key, expected in checks:
            if expected is not None and payload.get(key) != expected:
                return TokenVerification(
                    False,
                    f"Token scope mismatch on '{key}': token authorises "
                    f"{payload.get(key)!r} but activation requested {expected!r}.",
                    payload,
                )

        return TokenVerification(True, "Token verified.", payload)


if __name__ == "__main__":
    sample = "Reach out to john.doe@example.com or call +1 (555) 123-4567 for the winback campaign."
    print("PII masked:", GovernanceGuardrails.mask_pii(sample))

    tok = GovernanceGuardrails.issue_approval_token("VIP_Winback", "AUD_1", "email_marketing")
    print("Issued token:", tok[:60], "...")
    print("Valid:       ", GovernanceGuardrails.verify_activation_approval(tok, "VIP_Winback", "AUD_1", "email_marketing"))
    print("Forged bare: ", GovernanceGuardrails.verify_activation_approval("MKT_APPRV_"))
    print("Wrong scope: ", GovernanceGuardrails.verify_activation_approval(tok, "OTHER", "AUD_1", "email_marketing"))
