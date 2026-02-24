"""test_embedder.py — Standalone tests for the configured embedder model.

Tests:
    1. Connection: reachability of the embedder's base_url.
    2. Embedding: POST /v1/embeddings with a small test sentence and validate
       the response shape.
    3. Context window: POST /api/show (Ollama-specific) to retrieve the model's
       maximum context length from its metadata.

Usage:
    python test_embedder.py
"""

import json
import sys

import httpx

# ── Config ────────────────────────────────────────────────────────────────────
MODELS_FILE  = "llm_models_json.json"
TEST_SENTENCE = "Embed this sentence."

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"


def _load_embedder_cfg() -> tuple[str, dict]:
    """Load the first embedder entry from llm_models_json.json.

    Returns:
        (alias, cfg) tuple for the embedder model.

    Raises:
        SystemExit: If the file is missing or no embedder is configured.
    """
    try:
        with open(MODELS_FILE, "r", encoding="utf-8") as fh:
            models: dict[str, dict] = json.load(fh)
    except FileNotFoundError:
        print(f"[ERROR] '{MODELS_FILE}' not found. Run from the project root.")
        sys.exit(1)
    except json.JSONDecodeError as exc:
        print(f"[ERROR] Failed to parse '{MODELS_FILE}': {exc}")
        sys.exit(1)

    for alias, cfg in models.items():
        if cfg.get("kind") == "embedder":
            return alias, cfg

    print("[ERROR] No model with kind='embedder' found in llm_models_json.json.")
    sys.exit(1)


def _headers(cfg: dict) -> dict:
    """Build Authorization headers, skipping placeholder keys."""
    headers = {"Content-Type": "application/json"}
    api_key = cfg.get("api_key", "")
    if api_key and api_key.lower() not in ("", "none", "ollama", "no_key_required"):
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


# ── Test 1: Connection ────────────────────────────────────────────────────────

def test_connection(alias: str, cfg: dict) -> bool:
    """Check that the embedder's base_url is reachable (HTTP GET, any 2xx/3xx).

    Args:
        alias: Human-readable model alias for logging.
        cfg: Model configuration dict (must contain 'base_url').

    Returns:
        True if the host responds, False otherwise.
    """
    base_url = cfg["base_url"].rstrip("/")
    print(f"\n[Test 1] Connection  →  {base_url}")
    try:
        resp = httpx.get(base_url, timeout=10.0, follow_redirects=True)
        # Any HTTP response (even 404) means the host is up
        print(f"  HTTP {resp.status_code} received.")
        print(f"  {PASS}")
        return True
    except httpx.ConnectError as exc:
        print(f"  ConnectError: {exc}")
    except httpx.TimeoutException as exc:
        print(f"  TimeoutException: {exc}")
    except Exception as exc:  # noqa: BLE001
        print(f"  Unexpected error: {exc}")
    print(f"  {FAIL}")
    return False


# ── Test 2: Embedding ─────────────────────────────────────────────────────────

def test_embedding(alias: str, cfg: dict, sentence: str) -> bool:
    """POST /v1/embeddings and validate the response.

    Checks:
        - HTTP 200 response.
        - Response JSON contains a 'data' list with at least one item.
        - Each item has an 'embedding' key holding a non-empty list of floats.

    Args:
        alias: Model alias (used as the 'model' field in the request body).
        cfg: Model configuration dict.
        sentence: The text to embed.

    Returns:
        True if all checks pass, False otherwise.
    """
    model_name = cfg.get("model") or alias
    url = cfg["base_url"].rstrip("/") + "/v1/embeddings"
    body = {"model": model_name, "input": [sentence]}

    print(f"\n[Test 2] Embedding  →  POST {url}")
    print(f"  Model  : {model_name}")
    print(f"  Input  : \"{sentence}\"")

    try:
        resp = httpx.post(url, headers=_headers(cfg), json=body, timeout=30.0)
        resp.raise_for_status()
    except httpx.HTTPStatusError as exc:
        print(f"  HTTP error {exc.response.status_code}: {exc.response.text[:500]}")
        print(f"  {FAIL}")
        return False
    except (httpx.ConnectError, httpx.TimeoutException) as exc:
        print(f"  Request failed: {exc}")
        print(f"  {FAIL}")
        return False

    try:
        payload = resp.json()
    except Exception as exc:  # noqa: BLE001
        print(f"  Failed to parse JSON response: {exc}")
        print(f"  {FAIL}")
        return False

    # Validate response structure
    data = payload.get("data")
    if not isinstance(data, list) or len(data) == 0:
        print(f"  Unexpected response shape (no 'data' list): {payload}")
        print(f"  {FAIL}")
        return False

    embedding = data[0].get("embedding")
    if not isinstance(embedding, list) or len(embedding) == 0:
        print(f"  'embedding' key missing or empty in first data item.")
        print(f"  {FAIL}")
        return False

    # Spot-check that values are numeric
    if not all(isinstance(v, (int, float)) for v in embedding[:5]):
        print(f"  First 5 embedding values are not numeric: {embedding[:5]}")
        print(f"  {FAIL}")
        return False

    print(f"  Embedding dimension : {len(embedding)}")
    print(f"  First 5 values      : {[round(v, 6) for v in embedding[:5]]}")
    print(f"  {PASS}")
    return True


# ── Test 3: Context window ──────────────────────────────────────────────────

def test_context_window(alias: str, cfg: dict) -> bool:
    """Query the model's maximum context length via Ollama's POST /api/show.

    Ollama returns a 'modelinfo' dict whose keys are architecture-prefixed, e.g.
    'nomic-bert.context_length' or 'llama.context_length'. This test searches
    for any key ending in '.context_length' and reports its value.

    Note: This endpoint is Ollama-specific and will fail gracefully for other
    providers.

    Args:
        alias: Model alias for logging.
        cfg: Model configuration dict (must contain 'base_url' and 'model').

    Returns:
        True if the context length was found, False otherwise.
    """
    model_name = cfg.get("model") or alias
    url = cfg["base_url"].rstrip("/") + "/api/show"
    body = {"name": model_name}

    print(f"\n[Test 3] Context window  →  POST {url}")
    print(f"  Model : {model_name}")

    try:
        resp = httpx.post(url, json=body, timeout=15.0)
        resp.raise_for_status()
    except httpx.HTTPStatusError as exc:
        print(f"  HTTP error {exc.response.status_code}: {exc.response.text[:500]}")
        print(f"  (Endpoint may not be available for non-Ollama providers.)")
        print(f"  {FAIL}")
        return False
    except (httpx.ConnectError, httpx.TimeoutException) as exc:
        print(f"  Request failed: {exc}")
        print(f"  {FAIL}")
        return False

    try:
        payload = resp.json()
    except Exception as exc:  # noqa: BLE001
        print(f"  Failed to parse JSON response: {exc}")
        print(f"  {FAIL}")
        return False

    modelinfo: dict = payload.get("modelinfo", {})
    if not modelinfo:
        print(f"  No 'modelinfo' key in response. Full response: {payload}")
        print(f"  {FAIL}")
        return False

    # Search for any context_length key (architecture-prefixed)
    ctx_key = next(
        (k for k in modelinfo if k.endswith(".context_length")), None
    )
    if ctx_key is None:
        print(f"  No '*.context_length' key found in modelinfo.")
        print(f"  Available keys: {list(modelinfo.keys())}")
        print(f"  {FAIL}")
        return False

    context_length = modelinfo[ctx_key]
    print(f"  Context length key : {ctx_key}")
    print(f"  Max context window : {context_length:,} tokens")
    print(f"  {PASS}")
    return True


# ── Runner ────────────────────────────────────────────────────────────────────

def main() -> None:
    alias, cfg = _load_embedder_cfg()
    print(f"Embedder alias : '{alias}'")
    print(f"Model          : {cfg.get('model')}")
    print(f"Provider       : {cfg.get('provider')}")
    print(f"Base URL       : {cfg.get('base_url')}")

    results: list[bool] = []
    results.append(test_connection(alias, cfg))
    results.append(test_embedding(alias, cfg, TEST_SENTENCE))
    results.append(test_context_window(alias, cfg))

    passed = sum(results)
    total  = len(results)
    print(f"\n{'─' * 40}")
    print(f"Results: {passed}/{total} test(s) passed.")
    if passed < total:
        sys.exit(1)


if __name__ == "__main__":
    main()
