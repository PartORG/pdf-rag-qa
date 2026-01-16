"""Ollama LLM client for text generation."""

from typing import Optional, List, Dict, Any

import ollama
from ollama import ResponseError

from pdf_qa.core.exceptions import (
    OllamaConnectionError,
    ModelNotAvailableError,
    GenerationTimeoutError,
    LLMError,
)
from pdf_qa.core.logging import get_logger

logger = get_logger(__name__)


class OllamaLLM:
    """Client for Ollama LLM text generation."""

    def __init__(
        self,
        model_name: str = "llama3.2",
        host: str = "http://localhost:11434",
        timeout: int = 120,
    ):
        self.model_name = model_name
        self.host = host
        self.timeout = timeout
        self._client: Optional[ollama.Client] = None

    @property
    def client(self) -> ollama.Client:
        if self._client is None:
            self._client = ollama.Client(
                host=self.host,
                timeout=self.timeout,
            )
        return self._client

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Generate text response from prompt."""
        try:
            messages: List[Dict[str, str]] = []

            if system_prompt:
                messages.append({
                    "role": "system",
                    "content": system_prompt,
                })

            messages.append({
                "role": "user",
                "content": prompt,
            })

            options: Dict[str, Any] = {"temperature": temperature}
            if max_tokens:
                options["num_predict"] = max_tokens

            response = self.client.chat(
                model=self.model_name,
                messages=messages,
                options=options,
            )

            return response["message"]["content"]

        except ConnectionError:
            raise OllamaConnectionError(self.host)
        except ResponseError as e:
            if "not found" in str(e).lower():
                raise ModelNotAvailableError(self.model_name)
            raise LLMError(
                message="Generation failed",
                details=str(e),
            )
        except TimeoutError:
            raise GenerationTimeoutError(self.timeout)
        except Exception as e:
            raise LLMError(
                message="Unexpected generation error",
                details=str(e),
            )

    def is_available(self) -> bool:
        """Check if LLM model is available."""
        try:
            self.generate(prompt="Say hello.", max_tokens=5)
            return True
        except Exception:
            return False

    def list_models(self) -> List[str]:
        """List available models on Ollama server."""
        try:
            response = self.client.list()
            return [model["name"] for model in response.get("models", [])]
        except Exception:
            return []


def create_llm_from_settings() -> OllamaLLM:
    """Create OllamaLLM from settings."""
    from pdf_qa.config.settings import get_settings

    settings = get_settings()
    return OllamaLLM(
        model_name=settings.ollama_model,
        host=settings.ollama_host,
        timeout=settings.generation_timeout,
    )
