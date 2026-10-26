"""
llm.py — one small client for any OpenAI-compatible chat endpoint (Groq, Gemini).

Two jobs:
  generate_cleaning_script()  the model writes the pandas cleaning script from the
                              schema profile (never from the raw rows)
  explain_findings()          the model turns probe results into plain English

If no API key is configured, both fall back to deterministic templates and the
result is LABELLED `mode="fallback"`. Token counts are whatever the API reports
in its `usage` field; the fallback reports zero because nothing was sent.

Configure with environment variables (or a .env file in the repo root):
  RUBICON_LLM_PROVIDER = groq | gemini | none     (auto-detected from keys if unset)
  GROQ_API_KEY / GEMINI_API_KEY
  RUBICON_LLM_MODEL    = model name at that provider (optional; defaults below)
"""
import os
import re

import requests

PROVIDERS = {
    "groq": {"url": "https://api.groq.com/openai/v1/chat/completions",
             "key_env": "GROQ_API_KEY", "default_model": "openai/gpt-oss-120b"},
    "gemini": {"url": "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
               "key_env": "GEMINI_API_KEY", "default_model": "gemini-2.0-flash"},
}


def _load_dotenv():
    """Read .env from the repo root or the current directory. Real environment
    variables always win; quotes around values are tolerated."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for path in (os.path.join(root, ".env"), os.path.join(os.getcwd(), ".env")):
        if os.path.exists(path):
            for line in open(path):
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip().strip("\"'"))


def resolve_provider():
    _load_dotenv()
    p = os.getenv("RUBICON_LLM_PROVIDER", "").lower()
    if p in PROVIDERS and os.getenv(PROVIDERS[p]["key_env"]):
        return p
    for name, cfg in PROVIDERS.items():
        if os.getenv(cfg["key_env"]):
            return name
    return "none"


class LLMError(RuntimeError):
    pass


def chat(messages, provider, temperature=0.0, max_tokens=1500):
    cfg = PROVIDERS[provider]
    model = os.getenv("RUBICON_LLM_MODEL") or cfg["default_model"]
    r = requests.post(cfg["url"], timeout=90,
                      headers={"Authorization": f"Bearer {os.environ[cfg['key_env']]}",
                               "Content-Type": "application/json"},
                      json={"model": model, "messages": messages,
                            "temperature": temperature, "max_tokens": max_tokens})
    if r.status_code != 200:
        raise LLMError(f"{provider} returned {r.status_code}: {r.text[:500]}")
    data = r.json()
    usage = data.get("usage", {})
    return {"text": data["choices"][0]["message"]["content"], "model": model,
            "provider": provider,
            "usage": {"prompt_tokens": usage.get("prompt_tokens", 0),
                      "completion_tokens": usage.get("completion_tokens", 0)}}


# ------------------------------------------------------------- cleaning script
def fallback_cleaning_script(raw_path, clean_path, target, drop_cols):
    return "\n".join([
        "import pandas as pd",
        f"df = pd.read_csv({raw_path!r})",
        f"df = df.drop(columns=[c for c in {drop_cols!r} if c in df.columns])",
        "for c in df.columns:",
        f"    if c == {target!r}:",
        "        continue",
        "    if not pd.api.types.is_numeric_dtype(df[c]):",
        "        df[c] = df[c].fillna(df[c].mode().iloc[0] if df[c].notna().any() else 'missing')",
        "        df[c] = pd.factorize(df[c])[0]",
        "    else:",
        "        df[c] = df[c].fillna(df[c].median())",
        f"df.to_csv({clean_path!r}, index=False)",
        "print('clean shape', df.shape)",
    ])


CLEANING_PROMPT = """You are a careful data engineer. Write a short pandas script that
cleans a CSV for a binary classification task. You only see the schema profile below,
never the rows.

Rules:
1. Read {raw!r}; write the result to {clean!r} with index=False.
2. Drop these columns if present: {drop}.
3. Fill missing numeric values with the column median; missing text with the mode.
4. Encode every text column with pd.factorize (keep the same column name).
5. Do NOT modify, rename or drop any other column, including the target {target!r}
   and any id, time or treatment columns. Do not engineer new features.
6. Print the final shape. Use only pandas and numpy. Output ONLY a python code block.

Schema profile (JSON):
{profile}"""


def _extract_code(text):
    m = re.search(r"```(?:python)?\s*(.*?)```", text, re.S)
    return (m.group(1) if m else text).strip()


def generate_cleaning_script(prof, raw_path, clean_path, target, drop_cols):
    provider = resolve_provider()
    if provider == "none":
        return {"code": fallback_cleaning_script(raw_path, clean_path, target, drop_cols),
                "mode": "fallback", "provider": "none", "model": None,
                "usage": {"prompt_tokens": 0, "completion_tokens": 0}}
    prompt = CLEANING_PROMPT.format(raw=raw_path, clean=clean_path, drop=drop_cols,
                                    target=target, profile=prof)
    out = chat([{"role": "user", "content": prompt}], provider)
    return {"code": _extract_code(out["text"]), "mode": "llm", "provider": provider,
            "model": out["model"], "usage": out["usage"]}


# ------------------------------------------------------------- explanations
def explain_findings(rounds, plan):
    provider = resolve_provider()
    actions = [r.get("action") for r in rounds if r.get("action")]
    if provider == "none":
        if not actions:
            text = "No leakage probe fired; the validation score can be taken at face value."
        else:
            text = ("The audit found and fixed: " + "; ".join(actions) +
                    f". Final evaluation plan: {plan['cv']} cross-validation.")
        return {"text": text, "mode": "fallback",
                "usage": {"prompt_tokens": 0, "completion_tokens": 0}}
    summary = "\n".join(
        f"round {r['round']}: " + "; ".join(
            f"{p['probe']} score={p['score']} flagged={p['flagged']} ({p['evidence']})"
            for p in r["probes"]) + (f" -> action: {r['action']}" if r.get("action") else "")
        for r in rounds)
    prompt = ("Explain these data-leakage audit results to a product manager in at most "
              "four plain sentences. Name what was wrong, why it inflates model scores, "
              f"and what was done about it.\n\n{summary}\n\nFinal evaluation plan: {plan}")
    out = chat([{"role": "user", "content": prompt}], provider, max_tokens=400)
    return {"text": out["text"].strip(), "mode": "llm", "usage": out["usage"]}
