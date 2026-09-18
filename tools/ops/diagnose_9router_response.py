"""Print a sanitized summary of a 9router chat response (never prints credentials)."""

import asyncio
import json
import os

import httpx


async def main() -> None:
    url = os.environ["NINEROUTER_BASE_URL"].rstrip("/") + "/chat/completions"
    headers = {"Authorization": "Bearer " + os.getenv("NINEROUTER_API_KEY", "")}
    payload = {
        "model": os.environ["NINEROUTER_MODEL"],
        "messages": [{"role": "user", "content": 'Return only {"ok":true}'}],
        "response_format": {"type": "json_object"},
        "temperature": 0.2,
    }
    async with httpx.AsyncClient(timeout=120) as client:
        response = await client.post(url, headers=headers, json=payload)
    print(response.status_code, response.headers.get("content-type"))
    try:
        body = response.json()
        choice = (body.get("choices") or [{}])[0]
        print(json.dumps({
            "root_keys": list(body),
            "choice_keys": list(choice),
            "message": choice.get("message"),
            "finish_reason": choice.get("finish_reason"),
            "error": body.get("error"),
        }, ensure_ascii=False)[:4000])
    except Exception:
        print(response.text[:1000])


asyncio.run(main())

