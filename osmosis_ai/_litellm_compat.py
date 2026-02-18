# pyright: reportPrivateImportUsage=false
"""Thin re-export layer for litellm types.

pyright treats re-exports via a private-prefixed alias (_litellm) as private.
We suppress the diagnostic at file level so the rest of the codebase can import
cleanly.
"""

from __future__ import annotations

import litellm as _litellm

# ---------------------------------------------------------------------------
# Exception types
# ---------------------------------------------------------------------------
NotFoundError = _litellm.NotFoundError
APIError = _litellm.APIError
RateLimitError = _litellm.RateLimitError
AuthenticationError = _litellm.AuthenticationError
Timeout = _litellm.Timeout
APIConnectionError = _litellm.APIConnectionError
BudgetExceededError = _litellm.BudgetExceededError
ContextWindowExceededError = _litellm.ContextWindowExceededError

# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------
completion = _litellm.completion
acompletion = _litellm.acompletion
