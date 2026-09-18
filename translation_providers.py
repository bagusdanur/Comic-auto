"""External translation providers with strict structured-output validation."""

from __future__ import annotations

import abc
import asyncio
import json
import os
from typing import Any

import httpx
from language_memory import LANGUAGE_POLICY, terminology_errors, grounded_updates

SYSTEM_PROMPT = """You are a professional English to Indonesian comic translator.
Translate dialogue naturally into Indonesian. Do not translate word-for-word if it
sounds unnatural. Preserve emotion, character personality, jokes, insults, tone,
and context. Keep translations concise enough for comic speech bubbles. Do not add
explanations. Prefer natural colloquial Indonesian and contractions when they fit
the character's tone. Keep the visual length close to the source; do not expand a
short line into a formal long phrase. Do not censor ordinary dialogue. Return only valid structured JSON.
Maintain one register per speaker and scene: Indonesian teenagers default to
aku/kamu; use gue/lo only when surrounding dialogue clearly establishes that
rough register. Use saya/Anda when the relationship and situation call for formality.
Use familiar, easy-to-understand Indonesian rather than literary, archaic or
needlessly formal wording. For example, prefer pedagang over saudagar and
rombongan over karavan when the context permits. Translate idioms by their
intended meaning, not their individual words. Read neighboring lines together;
preserve interrupted sentences and never invent their endings. Silently check
each line for clarity, meaning, and conversational flow before returning it.
Do not force slang or simplify away emotion, threats, status or story facts.
Never change region IDs. The JSON object must contain a `translations` array."""

ADDRESS_POLICY = """Gendered forms of address:
- Never infer gender from the English word alone. Words such as teacher, doctor,
  boss, senior, and master are gender-neutral in English.
- Use Pak/Bapak or Bu/Ibu only when the speaker's gender is established by the
  supplied dialogue context, approved title memory, character name/pronouns, or
  reliable page-level visual context.
- If gender is not established, use a natural neutral form such as `Guru`, the
  character's name/title, or rewrite the sentence without a gendered honorific.
- Keep gendered pronouns and honorifics consistent across neighboring lines. Do
  not invent masculinity as a default."""

MEMORY_PROMPT = """Translation memory rules:
- Character names, aliases, organization names, named attacks, skills, abilities,
  named ranks, and named game/system terms must stay in their canonical English form unless
  the supplied memory explicitly specifies another form.
- Never translate a named skill literally. Preserve capitalization consistently.
- In game/system skill windows, preserve the RAW information hierarchy. Put the
  proper-name title, equipment/material type, rank, and DESCRIPTION on separate
  lines using `\n`; never merge the complete window into one paragraph.
- Generic occupations and organizations are not proper names: translate merchant
  as pedagang, mercenary as tentara bayaran, and merchant guild as serikat pedagang
  unless an approved title-specific mapping explicitly requires otherwise.
- Reuse every supplied memory mapping exactly.
- Also return a `memory_updates` object containing newly discovered reusable
  source-term to canonical-term mappings. Only include proper names and named
  terminology, never ordinary dialogue words."""

STYLE_PROMPT = """For every translation also return `style`, choosing exactly one:
`dialogue`, `narration`, `thought`, `shout`, or `sfx`. Use narration for caption or
exposition boxes, thought for internal monologue/inner voice, shout for screaming,
fear, panic, anger, or other strongly expressed speech, and sfx only for sound effects."""


class TranslationError(RuntimeError):
    pass


def parse_json_object(raw: str) -> dict[str, Any]:
    """Accept strict JSON plus common fenced JSON returned by compatible APIs."""
    text = raw.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            raise
        return json.loads(text[start:end + 1])


def parse_first_json_object(raw: str) -> dict[str, Any]:
    text = raw.strip()
    decoder = json.JSONDecoder()
    start = text.find("{")
    if start < 0:
        raise json.JSONDecodeError("No JSON object found", text, 0)
    payload, _ = decoder.raw_decode(text[start:])
    if not isinstance(payload, dict):
        raise json.JSONDecodeError("First JSON value is not an object", text, start)
    return payload


class TranslationProvider(abc.ABC):
    @abc.abstractmethod
    async def _request(self, prompt: str, schema: dict[str, Any]) -> str:
        raise NotImplementedError

    async def translate(
        self, items: list[dict], glossary: dict[str, str], memory_context: dict[str, Any] | None = None
    ) -> list[dict]:
        self.memory_updates: dict[str, str] = {}
        schema = {
            "type": "object",
            "properties": {
                "translations": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string"},
                            "translation": {"type": "string"},
                            "style": {"type": "string", "enum": ["dialogue", "narration", "thought", "shout", "sfx"]},
                        },
                        "required": ["id", "translation"],
                    },
                }
            },
            "required": ["translations"],
            "additionalProperties": True,
        }
        prompt = (
            SYSTEM_PROMPT
            + "\n\n" + MEMORY_PROMPT
            + "\n\n" + STYLE_PROMPT
            + "\n\n" + LANGUAGE_POLICY
            + "\n\n" + ADDRESS_POLICY
            + "\n\nGlossary (must be followed):\n"
            + json.dumps(glossary, ensure_ascii=False)
            + "\n\nApproved title-specific style/voice/examples (follow when relevant; terminology above wins):\n"
            + json.dumps(memory_context or {}, ensure_ascii=False)
            + "\n\nDialogue in reading order:\n"
            + json.dumps(items, ensure_ascii=False)
        )
        expected = [item["id"] for item in items]
        max_attempts = max(1, int(getattr(self, "max_attempts", 5)))
        last_error: Exception | None = None
        for attempt in range(1, max_attempts + 1):
            try:
                payload = parse_json_object(await self._request(prompt, schema))
                draft=payload.get('translations',[])
                if getattr(self, "single_pass", False):
                    edited_rows = draft
                else:
                    edited=parse_json_object(await self._request(
                        prompt+'\n\nYou are now the Indonesian dialogue editor. Revise this draft, checking every line against the original reading-order dialogue above. '
                        'Remove stiff calques and confusing phrasing, not just spelling errors. '
                        'For example, avoid "terlalu tidak tahu apa-apa"; express the contextual lack of understanding naturally. '
                        'Choose a natural form of address instead of literal occupational titles when appropriate. '
                        'Keep canonical names and terms, intensity, uncertainty, negation, comparisons and unfinished speech intact. '
                        'Do not add facts, flatten character voices or add trendy slang. Return the complete translations array with original IDs and styles.\n'
                        +json.dumps(draft,ensure_ascii=False),schema))
                    edited_rows = edited['translations']
                # The editorial pass occasionally paraphrases away a canonical
                # term that the first translation already preserved. Prefer the
                # glossary-safe draft instead of retrying the same bad edit until
                # the entire chapter is marked FAILED.
                if (
                    isinstance(draft, list)
                    and len(draft) == len(expected)
                    and isinstance(edited_rows, list)
                    and len(edited_rows) == len(expected)
                ):
                    for candidate_rows in (draft, edited_rows):
                        for row, expected_id in zip(candidate_rows, expected):
                            row["id"] = expected_id
                            if not row.get("translation"):
                                row["translation"] = row.get("translated_text") or row.get("text")
                    draft_issues = terminology_errors(items, draft, glossary)
                    edited_issues = terminology_errors(items, edited_rows, glossary)
                    payload['translations'] = draft if edited_issues and not draft_issues else edited_rows
                else:
                    payload['translations'] = edited_rows
                rows = payload["translations"]
                updates = payload.get("memory_updates", {})
                if isinstance(updates, dict):
                    self.memory_updates = {
                        str(source).strip(): str(target).strip()
                        for source, target in updates.items()
                        if isinstance(source, str) and isinstance(target, str)
                        and source.strip() and target.strip()
                        and len(source.strip()) <= 120 and len(target.strip()) <= 120
                    }
                for row in rows:
                    if not row.get("translation"):
                        row["translation"] = row.get("translated_text") or row.get("text")
                ids = [row["id"] for row in rows]
                if ids != expected:
                    # Structured-output models sometimes renumber otherwise
                    # correctly ordered rows. Count/order is enough to repair
                    # this safely; missing or extra rows still fail and retry.
                    if len(rows) == len(expected):
                        for row, expected_id in zip(rows, expected):
                            row["id"] = expected_id
                    else:
                        raise TranslationError(f"Provider changed IDs/order: {ids}")
                empty_ids = [
                    row.get("id", "<missing>") for row in rows
                    if not isinstance(row.get("translation"), str) or not row["translation"].strip()
                ]
                if empty_ids:
                    raise TranslationError(f"Provider returned empty translations: {empty_ids}")
                issues=terminology_errors(items,rows,glossary)
                if issues:
                    prompt += '\n\nCorrect these terminology violations in the complete JSON response:\n' + '\n'.join(issues)
                    raise TranslationError('; '.join(issues))
                self.memory_updates=grounded_updates(self.memory_updates,items,glossary)
                return rows
            except (KeyError, TypeError, ValueError, httpx.HTTPError, TranslationError) as error:
                last_error = error
                if attempt < max_attempts:
                    await asyncio.sleep(attempt)
        raise TranslationError(f"Translation failed after {max_attempts} attempts: {last_error}")


class DeepSeekProvider(TranslationProvider):
    def __init__(self) -> None:
        self.api_key = os.environ["DEEPSEEK_API_KEY"]
        self.model = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
        self.base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com").rstrip("/")

    async def _request(self, prompt: str, schema: dict[str, Any]) -> str:
        del schema
        async with httpx.AsyncClient(timeout=90) as client:
            response = await client.post(
                f"{self.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    "response_format": {"type": "json_object"},
                    "temperature": 0.2,
                },
            )
            response.raise_for_status()
            try:
                payload = response.json()
                content = payload["choices"][0]["message"]["content"]
            except (json.JSONDecodeError, KeyError, IndexError, TypeError) as error:
                preview = response.text.strip()[:180] or "<empty body>"
                raise TranslationError(f"Provider returned an invalid response: {preview}") from error
            if not isinstance(content, str) or not content.strip():
                raise TranslationError("9router returned empty message content")
            return content


class GeminiProvider(TranslationProvider):
    def __init__(self) -> None:
        self.api_key = os.getenv("GEMINI_API_KEY") or os.environ["GOOGLE_API_KEY"]
        self.model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

    async def _request(self, prompt: str, schema: dict[str, Any]) -> str:
        async with httpx.AsyncClient(timeout=90) as client:
            response = await client.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent",
                headers={"x-goog-api-key": self.api_key},
                json={
                    "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
                    "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                    "generationConfig": {
                        "temperature": 0.2,
                        "responseMimeType": "application/json",
                        "responseSchema": schema,
                    },
                },
            )
            response.raise_for_status()
            return response.json()["candidates"][0]["content"]["parts"][0]["text"]


class OpenAICompatibleProvider(TranslationProvider):
    def __init__(self) -> None:
        self.api_key = os.getenv("OPENAI_COMPATIBLE_API_KEY") or os.environ["XIAOMI_API_KEY"]
        self.model = os.getenv("OPENAI_COMPATIBLE_MODEL", "mimo-v2.5")
        self.base_url = os.getenv(
            "OPENAI_COMPATIBLE_BASE_URL", "https://api.xiaomimimo.com/v1"
        ).rstrip("/")

    async def _request(self, prompt: str, schema: dict[str, Any]) -> str:
        del schema
        async with httpx.AsyncClient(timeout=90) as client:
            response = await client.post(
                f"{self.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    "response_format": {"type": "json_object"},
                    "temperature": 0.2,
                },
            )
            response.raise_for_status()
            try:
                payload = response.json()
                content = payload["choices"][0]["message"]["content"]
            except (json.JSONDecodeError, KeyError, IndexError, TypeError) as error:
                preview = response.text.strip()[:180] or "<empty body>"
                raise TranslationError(f"9router returned an invalid response: {preview}") from error
            if not isinstance(content, str) or not content.strip():
                raise TranslationError("9router returned empty message content")
            return content


class NineRouterProvider(TranslationProvider):
    """OpenAI-compatible provider served locally by 9router on the VPS."""

    def __init__(self) -> None:
        self.api_key = os.getenv("NINEROUTER_API_KEY", "")
        self.model = os.getenv("NINEROUTER_MODEL", "ag/gemini-3.5-flash-low")
        self.single_pass = os.getenv("NINEROUTER_SINGLE_PASS", "1").lower() not in {"0", "false", "no"}
        self.max_attempts = int(os.getenv("NINEROUTER_MAX_ATTEMPTS", "1"))
        self.fallback_models = [
            item.strip() for item in os.getenv(
                "NINEROUTER_FALLBACK_MODELS",
                (
                    "ag/claude-sonnet-4-6,"
                    "glm/GLM-5.3-Flash,"
                    "glm/glm-4.7"
                ),
            ).split(",") if item.strip() and item.strip() != self.model
        ]
        self.base_url = os.getenv(
            "NINEROUTER_BASE_URL", "http://127.0.0.1:20128/v1"
        ).rstrip("/")

    async def _request(self, prompt: str, schema: dict[str, Any]) -> str:
        del schema
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        timeout = float(os.getenv("NINEROUTER_TIMEOUT_SECONDS", "60"))
        async with httpx.AsyncClient(timeout=timeout) as client:
            last_error: Exception | None = None
            for model in [self.model, *self.fallback_models]:
                try:
                    payload: dict[str, Any] = {}
                    message: dict[str, Any] = {}
                    request_payload = {
                                "model": model,
                                "messages": [
                                    {"role": "system", "content": SYSTEM_PROMPT},
                                    {"role": "user", "content": prompt},
                                ],
                                "response_format": {"type": "json_object"},
                                "temperature": 0.2,
                    }
                    response = await asyncio.wait_for(
                        client.post(
                            f"{self.base_url}/chat/completions",
                            headers=headers,
                            json=request_payload,
                        ),
                        timeout=timeout + 5,
                    )
                    # Move through the ordered provider chain only for
                    # conditions another upstream model can actually heal.
                    if response.status_code in (404, 429, 500, 502, 503, 504):
                        response.raise_for_status()
                    else:
                        response.raise_for_status()
                    is_sse = (
                        "text/event-stream" in response.headers.get("content-type", "")
                        and any(line.startswith("data: ") for line in response.text.splitlines())
                    )
                    if is_sse:
                        chunks: list[str] = []
                        for line in response.text.splitlines():
                            if not line.startswith("data: ") or line == "data: [DONE]":
                                continue
                            event = json.loads(line[6:])
                            delta = event.get("choices", [{}])[0].get("delta", {})
                            if isinstance(delta.get("content"), str):
                                chunks.append(delta["content"])
                        content = "".join(chunks)
                    else:
                        try:
                            payload = response.json()
                        except json.JSONDecodeError:
                            payload = parse_first_json_object(response.text)
                        message = payload["choices"][0]["message"]
                        content = self._message_text(message)
                    if not isinstance(content, str) or not content.strip():
                        request_payload.pop("response_format", None)
                        response = await asyncio.wait_for(
                            client.post(
                                f"{self.base_url}/chat/completions",
                                headers=headers,
                                json=request_payload,
                            ),
                            timeout=timeout + 5,
                        )
                        response.raise_for_status()
                        is_sse = (
                            "text/event-stream" in response.headers.get("content-type", "")
                            and any(line.startswith("data: ") for line in response.text.splitlines())
                        )
                        if is_sse:
                            chunks = []
                            for line in response.text.splitlines():
                                if not line.startswith("data: ") or line == "data: [DONE]":
                                    continue
                                event = json.loads(line[6:])
                                delta = event.get("choices", [{}])[0].get("delta", {})
                                if isinstance(delta.get("content"), str):
                                    chunks.append(delta["content"])
                            content = "".join(chunks)
                        else:
                            try:
                                payload = response.json()
                            except json.JSONDecodeError:
                                payload = parse_first_json_object(response.text)
                            message = payload["choices"][0]["message"]
                            content = self._message_text(message)
                    if not isinstance(content, str) or not content.strip():
                        detail = ",".join(sorted(str(key) for key in message.keys())) if isinstance(message, dict) else type(message).__name__
                        finish = payload.get("choices", [{}])[0].get("finish_reason") if isinstance(payload, dict) else None
                        response_preview = response.text.strip().replace("\n", " ")[:320]
                        raise TranslationError(
                            f"9router returned empty content from {model} "
                            f"(message keys: {detail or '<none>'}; finish_reason: {finish!r}; "
                            f"content-type: {response.headers.get('content-type', '<none>')}; body: {response_preview!r})"
                        )
                    if "is no longer available" in content.lower():
                        raise TranslationError(f"9router model unavailable from {model}: {content.strip()}")
                    return content
                except (asyncio.TimeoutError, json.JSONDecodeError, KeyError, IndexError, TypeError, httpx.HTTPError, TranslationError) as error:
                    last_error = error
                    continue
            raise TranslationError(f"9router failed across fallback models: {last_error}")

    @staticmethod
    def _message_text(message: Any) -> str:
        """Normalize OpenAI-compatible text variants emitted by 9router upstreams."""
        if not isinstance(message, dict):
            return ""
        for key in ("content", "reasoning_content", "reasoning", "text", "output_text"):
            value = message.get(key)
            if isinstance(value, str) and value.strip():
                return value
            if isinstance(value, list):
                parts: list[str] = []
                for item in value:
                    if isinstance(item, str):
                        parts.append(item)
                    elif isinstance(item, dict):
                        text = item.get("text") or item.get("content") or item.get("value")
                        if isinstance(text, str):
                            parts.append(text)
                joined = "".join(parts)
                if joined.strip():
                    return joined
        return ""


def create_provider(name: str) -> TranslationProvider:
    if name == "deepseek":
        return DeepSeekProvider()
    if name == "gemini":
        return GeminiProvider()
    if name == "openai-compatible":
        return OpenAICompatibleProvider()
    if name == "9router":
        return NineRouterProvider()
    raise ValueError(f"Unknown provider: {name}")
