from typing import Protocol


class TeacherConfigError(Exception):
    pass


class Teacher(Protocol):
    def chat(self, messages: list[dict], temperature: float | None = None) -> str: ...


class OpenAITeacher:
    def __init__(self, base_url: str, model: str, api_key: str, temperature: float = 0.8):
        from openai import OpenAI
        self.model = model
        self.temperature = temperature
        self._client = OpenAI(base_url=base_url, api_key=api_key or "not-needed")

    def chat(self, messages: list[dict], temperature: float | None = None) -> str:
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature if temperature is None else temperature,
        )
        return resp.choices[0].message.content


class FakeTeacher:
    def __init__(self, responses: list[str]):
        self._responses = list(responses)
        self._i = 0
        self.calls: list[list[dict]] = []

    def chat(self, messages: list[dict], temperature: float | None = None) -> str:
        self.calls.append(messages)
        r = self._responses[self._i % len(self._responses)]
        self._i += 1
        return r


def from_config(cfg: dict) -> OpenAITeacher:
    t = cfg["teacher"]
    if not t.get("base_url"):
        raise TeacherConfigError("teacher.base_url is required")
    return OpenAITeacher(
        base_url=t["base_url"], model=t["model"],
        api_key=t.get("api_key", ""), temperature=t.get("temperature", 0.8),
    )
