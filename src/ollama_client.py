"""
Thin wrapper around the Ollama REST API.
All scripts import from here so the base URL is configured in one place.
"""
import os
import json
import requests
from dotenv import load_dotenv

load_dotenv()

BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
DEFAULT_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")


def generate(
    prompt: str,
    model: str = DEFAULT_MODEL,
    system: str = "",
    temperature: float = 0.7,
    stream: bool = False,
) -> str:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": stream,
        "options": {"temperature": temperature},
    }
    if system:
        payload["system"] = system

    if stream:
        response = requests.post(f"{BASE_URL}/api/generate", json=payload, stream=True)
        response.raise_for_status()
        result = ""
        for line in response.iter_lines():
            if line:
                chunk = json.loads(line)
                token = chunk.get("response", "")
                print(token, end="", flush=True)
                result += token
                if chunk.get("done"):
                    break
        print()  # newline after streaming finishes
        return result
    else:
        response = requests.post(f"{BASE_URL}/api/generate", json=payload)
        response.raise_for_status()
        return response.json()["response"]


def embed(text: str, model: str = DEFAULT_EMBED_MODEL) -> list[float]:
    response = requests.post(
        f"{BASE_URL}/api/embed",
        json={"model": model, "input": text},
    )
    response.raise_for_status()
    return response.json()["embeddings"][0]


def embed_batch(texts: list[str], model: str = DEFAULT_EMBED_MODEL) -> list[list[float]]:
    return [embed(t, model) for t in texts]


def list_models() -> list[str]:
    response = requests.get(f"{BASE_URL}/api/tags")
    response.raise_for_status()
    return [m["name"] for m in response.json().get("models", [])]
