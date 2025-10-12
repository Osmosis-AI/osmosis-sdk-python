# Provider Integrations

Provider-specific integrations for `evaluate_rubric` live in this package. Each module
subclasses `RubricProvider`, implements the request/response flow for a hosted LLM,
and registers itself with the global provider registry so that
`osmosis_ai.rubric_eval.evaluate_rubric` can route requests by provider name.

## Directory Structure

- `base.py` — Core abstractions such as `RubricProvider`, `ProviderRequest`, and the
  registry that tracks available providers.
- `shared.py` — Utility helpers (debug logging, schema definitions, JSON sanitising)
  reused across provider implementations.
- `openai_family.py`, `anthropic_provider.py`, `gemini_provider.py` — Built-in
  providers that wrap the official vendor SDKs (OpenAI and xAI share the OpenAI client wrapper).
- `__init__.py` — Wires the default providers into the registry and exposes
  `register_provider`, `get_provider`, and `supported_providers`.

## Adding a Provider

1. **Create a provider class**
   ```python
   # my_provider.py
   from osmosis_ai.providers.base import ProviderRequest, RubricProvider
   from osmosis_ai.rubric_types import RewardRubricRunResult

   class MyProvider(RubricProvider):
       name = "my-provider"

       def default_timeout(self, model: str) -> float:
           return 30.0  # optional override

       def run(self, request: ProviderRequest) -> RewardRubricRunResult:
           # Call the vendor SDK with request.system_content / request.user_content
           # Return {"score": float, "explanation": str, "raw": <original payload>}
           ...
   ```
   The `ProviderRequest` dataclass supplies the prompt, score range, timeout, and a
   request identifier for logging or tracing.

2. **Handle optional dependencies gracefully**  
   Wrap imports in `try/except ImportError` and raise a `RuntimeError` with install
   instructions when the SDK is missing (see the existing providers for the pattern).

3. **Register the provider**  
   Add the provider to the global registry during application start-up:
   ```python
   from osmosis_ai.providers import register_provider
   from .my_provider import MyProvider

   register_provider(MyProvider())
   ```
   This can live in your package’s `__init__` or a dedicated bootstrap module.

4. **Invoke `evaluate_rubric`**  
   Users can now call:
   ```python
   from osmosis_ai import evaluate_rubric

   evaluate_rubric(
       rubric="...",
       messages=[...],
       model_info={"provider": "my-provider", "model": "..."},
   )
   ```
   API keys are resolved via `model_info["api_key"]` or
   `model_info["api_key_env"]`. If neither is provided, `evaluate_rubric` looks up the
   default mapping defined in `DEFAULT_API_KEY_ENV` (extend it if needed). You can also
   provide rubric-specific overrides through the typed `ModelInfo`, including `score_min`,
   `score_max`, `system_prompt`, `original_input`, and `timeout`.

5. **Document usage**  
   Update your README or onboarding material with the provider name, model IDs, and
   environment variables required for authentication.

## Tips

- Reuse helpers from `shared.py` for JSON schema definitions, logging, and response
  sanitisation to keep behaviour consistent.
- Honour `request.timeout` and `request.score_min`/`score_max` to ensure predictable
  behaviour across providers.
