"""LLM 호출 경계.

transformers + bitsandbytes 를 프로세스 안에서 직접 돌리지 않고 Ollama(HTTP)에 맡긴다.
이전 구현이 좌초한 지점이 정확히 trust_remote_code × transformers 버전 충돌이었고,
그건 모델을 바꿔도 반복된다. HTTP 한 겹을 사이에 두면 모델 교체가 설정 한 줄이 되고,
앱 컨테이너에서 CUDA 스택이 통째로 빠진다.

다른 백엔드를 붙이고 싶으면 LLMClient 만 구현하면 된다 — pipeline 은 이 프로토콜만 안다.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any, Protocol

import httpx


class LLMError(RuntimeError):
    """LLM 을 부를 수 없거나 응답이 형식에 맞지 않을 때."""


class LLMClient(Protocol):
    def complete(self, system: str, user: str) -> str: ...

    def stream(self, system: str, user: str) -> Iterator[str]: ...

    def health(self) -> bool: ...


class OllamaClient:
    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        temperature: float = 0.1,
        max_tokens: int = 512,
        timeout_s: float = 120.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        # 스트리밍은 첫 토큰까지만 빠르면 되고 그 뒤로는 계속 흘러오므로,
        # 연결·읽기 타임아웃을 나눠 잡는다. 단일 read 타임아웃이면 긴 생성이 끊긴다.
        self._client = httpx.Client(timeout=httpx.Timeout(timeout_s, connect=10.0))

    def close(self) -> None:
        self._client.close()

    def health(self) -> bool:
        """서버가 살아 있고 해당 모델이 받아져 있는지. 둘을 구분하지 않고 bool 로만 답한다."""
        try:
            response = self._client.get(f"{self.base_url}/api/tags", timeout=5.0)
            response.raise_for_status()
        except httpx.HTTPError:
            return False
        names = {m.get("name", "") for m in response.json().get("models", [])}
        # Ollama 는 태그를 생략하면 :latest 로 채워 돌려주므로 접두사로 비교한다.
        return any(n == self.model or n.startswith(f"{self.model.split(':')[0]}:") for n in names)

    def _payload(self, system: str, user: str, *, stream: bool) -> dict[str, Any]:
        return {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": stream,
            "options": {"temperature": self.temperature, "num_predict": self.max_tokens},
        }

    def _fail(self, exc: Exception) -> LLMError:
        return LLMError(f"Ollama 호출 실패({self.base_url}, model={self.model}): {exc}")

    def complete(self, system: str, user: str) -> str:
        try:
            response = self._client.post(
                f"{self.base_url}/api/chat", json=self._payload(system, user, stream=False)
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise self._fail(exc) from exc

        content = response.json().get("message", {}).get("content")
        if not isinstance(content, str) or not content.strip():
            raise LLMError("Ollama 응답에 message.content 가 없습니다")
        return content.strip()

    def stream(self, system: str, user: str) -> Iterator[str]:
        """생성되는 대로 조각을 흘려보낸다.

        체감 속도를 지배하는 건 전체 생성 시간이 아니라 첫 토큰까지의 시간이다.
        8B 모델에 4천자 프롬프트면 완성까지 십수 초가 걸리지만, 스트리밍이면
        1초 안에 글자가 뜨기 시작한다.

        Ollama 의 스트리밍 응답은 SSE 가 아니라 줄 단위 JSON(NDJSON)이다.
        """
        try:
            with self._client.stream(
                "POST", f"{self.base_url}/api/chat", json=self._payload(system, user, stream=True)
            ) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if not line:
                        continue
                    try:
                        message = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise LLMError(
                            f"스트림에서 JSON 이 아닌 줄을 받았습니다: {line[:120]}"
                        ) from exc
                    if message.get("error"):
                        raise LLMError(str(message["error"]))
                    piece = message.get("message", {}).get("content")
                    if piece:
                        yield piece
                    if message.get("done"):
                        return
        except httpx.HTTPError as exc:
            raise self._fail(exc) from exc
