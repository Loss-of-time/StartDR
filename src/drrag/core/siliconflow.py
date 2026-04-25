"""硅基流动 OpenAI 兼容接口封装。"""

import json
import os
from dataclasses import dataclass
from typing import cast
from urllib.request import Request, urlopen

from .schema import RagGenerationUsage
from .setting import DEFAULT_SILICONFLOW_CHAT_COMPLETIONS_URL


@dataclass(slots=True)
class SiliconFlowCompletion:
    """单次硅基流动对话请求结果。"""

    raw_response: dict[str, object]
    response_content: str
    finish_reason: str | None
    trace_id: str | None
    usage: RagGenerationUsage | None


def read_api_key() -> str:
    """读取硅基流动 API Key。"""

    return os.environ["SILICONFLOW_API_KEY"]


def request_json_completion(
    model_name: str,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int,
    temperature: float,
    timeout_seconds: int,
) -> SiliconFlowCompletion:
    """调用硅基流动 JSON Mode 完成一次结构化生成。"""

    request_body: dict[str, object] = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "n": 1,
        # 目的：强制模型返回 JSON 字符串，避免药品解释输出漂移为自然语言段落。
        "response_format": {"type": "json_object"},
    }
    http_request: Request = Request(
        url=DEFAULT_SILICONFLOW_CHAT_COMPLETIONS_URL,
        data=json.dumps(request_body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {read_api_key()}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urlopen(http_request, timeout=timeout_seconds) as response:
        response_text: str = response.read().decode("utf-8")
        trace_id: str | None = response.headers.get("x-siliconcloud-trace-id")
    raw_response: dict[str, object] = cast(dict[str, object], json.loads(response_text))
    choices: list[object] = cast(list[object], raw_response["choices"])
    first_choice: dict[str, object] = cast(dict[str, object], choices[0])
    message: dict[str, object] = cast(dict[str, object], first_choice["message"])
    usage_payload: dict[str, object] | None = cast(
        dict[str, object] | None, raw_response.get("usage")
    )
    usage: RagGenerationUsage | None = (
        RagGenerationUsage(
            prompt_tokens=cast(int | None, usage_payload.get("prompt_tokens")),
            completion_tokens=cast(int | None, usage_payload.get("completion_tokens")),
            total_tokens=cast(int | None, usage_payload.get("total_tokens")),
        )
        if usage_payload is not None
        else None
    )
    return SiliconFlowCompletion(
        raw_response=raw_response,
        response_content=cast(str, message["content"]),
        finish_reason=cast(str | None, first_choice.get("finish_reason")),
        trace_id=trace_id,
        usage=usage,
    )
