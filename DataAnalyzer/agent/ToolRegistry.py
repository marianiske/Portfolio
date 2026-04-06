from typing import Any, Callable

class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, dict[str, Any]] = {}

    def register(
        self,
        name: str,
        description: str,
        parameters: dict[str, Any],
        func: Callable[..., dict[str, Any]],
    ) -> None:
        self._tools[name] = {
            "description": description,
            "parameters": parameters,
            "func": func,
        }

    def ollama_tools(self) -> list[dict[str, Any]]:
        tools = []
        for name, spec in self._tools.items():
            tools.append({
                "type": "function",
                "function": {
                    "name": name,
                    "description": spec["description"],
                    "parameters": spec["parameters"],
                },
            })
        return tools

    def execute(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if name not in self._tools:
            raise KeyError(f"Unbekanntes Tool: {name}")

        func = self._tools[name]["func"]
        return func(**arguments)
