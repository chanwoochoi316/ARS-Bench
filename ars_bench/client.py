"""OpenAI-compatible HTTP without a required SDK; no credentials are persisted."""
import json
import os
import time
import urllib.error
import urllib.request


def post(url, body, headers=None, timeout=180, retries=3):
    request = urllib.request.Request(url, json.dumps(body).encode(),
                                     {"Content-Type": "application/json", **(headers or {})})
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.load(response)
        except urllib.error.HTTPError as exc:
            if exc.code not in (408, 429, 500, 502, 503, 504) or attempt == retries:
                raise RuntimeError(f"HTTP {exc.code} from configured service") from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            if attempt == retries:
                raise RuntimeError("Configured service could not be reached") from exc
        time.sleep(min(2 ** attempt, 10))


class Client:
    def __init__(self, base_url, model, key_env="OPENAI_API_KEY", seed=1, timeout=180):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.key_env = key_env
        self.seed = seed
        self.timeout = timeout

    def chat(self, messages, temperature=0, max_tokens=4096, **options):
        body = {"model": self.model, "messages": messages, "temperature": temperature,
                "max_tokens": max_tokens, "seed": self.seed, **options}
        headers = {}
        key = os.environ.get(self.key_env)
        if key:
            headers["Authorization"] = f"Bearer {key}"
        result = post(self.base_url + "/chat/completions", body, headers, self.timeout)
        message = result["choices"][0]["message"]
        content = message.get("content") or ""
        reasoning = message.get("reasoning_content") or message.get("reasoning") or ""
        if reasoning:
            content = f"<think>{reasoning}</think>\n{content}"
        return content

    def complete(self, prompt, **options):
        body = {"model": self.model, "prompt": prompt, "seed": self.seed, **options}
        headers = {}
        key = os.environ.get(self.key_env)
        if key:
            headers["Authorization"] = f"Bearer {key}"
        result = post(self.base_url + "/completions", body, headers, self.timeout)
        return result["choices"][0]["text"]


def parse_object(text):
    import re
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        value = json.loads(text[start:end + 1])
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object from model")
    return value
