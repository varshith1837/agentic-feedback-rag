"""Minimal, testable Groq/Ollama JSON clients with caching and retry controls.

No key is stored in the repo, database, or request logs.
"""
from __future__ import annotations
import hashlib
import json
import os
import time
import httpx
from .config import load_dotenv

def extract_retry_after(response: httpx.Response, fallback_delay: float) -> float:
    header_val = response.headers.get('retry-after')
    if header_val:
        try:
            return max(float(header_val), 1.0)
        except ValueError:
            pass
    reset_tokens = response.headers.get('x-ratelimit-reset-tokens')
    if reset_tokens:
        try:
            return max(float(reset_tokens.rstrip('s')), 1.0)
        except ValueError:
            pass
    try:
        data = response.json()
        msg = str(data.get('error', {}).get('message', ''))
        import re
        m = re.search(r'try again in ([\d\.]+)s', msg)
        if m:
            return max(float(m.group(1)), 1.0)
        m = re.search(r'try again in ([\d\.]+)ms', msg)
        if m:
            return max(float(m.group(1)) / 1000.0, 1.0)
    except Exception:
        pass
    return fallback_delay

class LLMError(RuntimeError):
    pass

class LLMClient:
    def __init__(self, provider: str = 'ollama', model: str | None = None, timeout: int = 120):
        load_dotenv()
        if provider not in ('groq', 'ollama'):
            raise ValueError('provider must be groq or ollama')
        self.provider = provider
        self.model = model or (os.getenv('GROQ_MODEL', 'llama-3.3-70b-versatile') if provider == 'groq'
                               else os.getenv('OLLAMA_MODEL', 'qwen2.5:7b'))
        self.timeout = timeout
        if provider == 'groq' and not os.getenv('GROQ_API_KEY'):
            raise LLMError('GROQ_API_KEY not configured. Set it in .env, or use --provider rules.')
        self.calls = 0

    def request_key(self, messages: list[dict], max_tokens: int = 1600) -> str:
        ident = json.dumps({'provider':self.provider, 'model':self.model,
                            'messages':messages, 'max_tokens':max_tokens}, sort_keys=True,ensure_ascii=False)
        return hashlib.sha256(ident.encode()).hexdigest()

    def invalidate(self, db, messages: list[dict], max_tokens: int = 1600):
        db.execute('DELETE FROM llm_cache WHERE request_hash=?', (self.request_key(messages,max_tokens),))
        db.commit()

    def json(self, messages: list[dict], db=None, max_tokens: int = 1600) -> dict:
        key = self.request_key(messages,max_tokens)
        if db is not None:
            cached = db.one('SELECT response_json FROM llm_cache WHERE request_hash=?', (key,))
            if cached is not None:
                return json.loads(cached['response_json'])
        if self.provider == 'groq':
            url = 'https://api.groq.com/openai/v1/chat/completions'
            headers = {'Authorization': 'Bearer ' + os.environ['GROQ_API_KEY']}
            payload = {'model':self.model, 'messages':messages,
                       'temperature':0, 'response_format':{'type':'json_object'},
                       'max_tokens':max_tokens}
        else:
            url = os.getenv('OLLAMA_URL','http://localhost:11434').rstrip('/') + '/api/chat'
            headers = {}
            payload = {'model':self.model, 'messages':messages, 'stream':False,
                       'format':'json', 'options':{'temperature':0, 'num_predict':max_tokens}}
        max_attempts = 15
        delay = 1.0
        last_error = None
        for attempt in range(max_attempts):
            try:
                response = httpx.post(url, json=payload, headers=headers, timeout=self.timeout)
                if response.status_code == 429:
                    wait_sec = extract_retry_after(response, fallback_delay=min(delay * 2, 30.0)) + 1.0
                    last_error = f'HTTP 429: {response.text[:300]}'
                    if attempt < max_attempts - 1:
                        time.sleep(wait_sec)
                        delay = min(delay * 2, 30.0)
                        continue
                elif response.status_code in (500, 502, 503, 504):
                    last_error = f'HTTP {response.status_code}: {response.text[:300]}'
                    if attempt < max_attempts - 1:
                        time.sleep(min(delay, 20.0))
                        delay *= 2
                        continue
                response.raise_for_status()
                body = response.json()
                content = (body['choices'][0]['message']['content'] if self.provider == 'groq'
                           else body['message']['content'])
                result = parse_json(content)
                self.calls += 1
                if db is not None:
                    # Cache validated JSON payload only; no secret headers.
                    db.execute('INSERT INTO llm_cache(request_hash,response_json) VALUES (?,?)',
                               (key, json.dumps(result, ensure_ascii=False)))
                    db.commit()
                return result
            except httpx.HTTPStatusError as e:
                last_error = str(e)
                if e.response.status_code == 429:
                    wait_sec = extract_retry_after(e.response, fallback_delay=min(delay * 2, 30.0)) + 1.0
                    if attempt < max_attempts - 1:
                        time.sleep(wait_sec)
                        delay = min(delay * 2, 30.0)
                        continue
                # Invalid model, wrong key, bad request: do not wait and retry.
                if e.response.status_code not in (429,500,502,503,504):
                    raise LLMError(f'{self.provider} HTTP {e.response.status_code}: {e.response.text[:250]}') from e
                if attempt < max_attempts - 1:
                    time.sleep(min(delay,20.0))
                    delay *= 2
            except (httpx.HTTPError, ValueError, KeyError, TypeError) as e:
                last_error = str(e)
                if attempt < max_attempts - 1:
                    time.sleep(min(delay, 20.0))
                    delay *= 2
        raise LLMError(f'{self.provider} request failed: {last_error}')

def parse_json(content: str | dict) -> dict:
    if isinstance(content, dict):
        return content
    text = content.strip()
    if text.startswith('```'):
        parts = text.split('\n',1)
        text = parts[1].rsplit('```',1)[0].strip() if len(parts)>1 else text
    decoded = json.loads(text)
    if not isinstance(decoded, dict):
        raise ValueError('LLM output must be a JSON object')
    return decoded
