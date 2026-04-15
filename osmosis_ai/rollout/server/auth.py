from dataclasses import dataclass, field


@dataclass
class ControllerAuth:
    api_key: str | None = field(default=None, repr=False)

    def __repr__(self) -> str:
        return (
            "ControllerAuth(api_key=<redacted>)"
            if self.api_key
            else "ControllerAuth(api_key=None)"
        )

    def as_bearer_headers(self) -> dict[str, str] | None:
        if not self.api_key:
            return None
        return {"Authorization": f"Bearer {self.api_key}"}
