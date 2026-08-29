import base64
import os
from typing import Any

import requests

from app.core.config import settings
from app.core.logging import logger
from app.services.base import BasePlateRecognizer
from app.services.constants import INDIAN_PLATE_REGEX


def extract_message_content(payload: Any) -> str | None:
    """
    Pull the assistant text out of an OpenAI-shaped chat completion response.

    Safely traverses the nested dictionary structure (choices -> message -> content)
    with explicit type checks at each level to handle rate limits, content filters,
    or malformed responses without raising unhandled KeyErrors/IndexErrors.
    """
    if not isinstance(payload, dict):
        return None

    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return None

    first = choices[0]
    if not isinstance(first, dict):
        return None

    message = first.get("message")
    if not isinstance(message, dict):
        return None

    content = message.get("content")
    if not isinstance(content, str):
        return None

    stripped = content.strip()
    return stripped or None


class NvidiaVisionStrategy(BasePlateRecognizer):
    """
    Concrete Strategy using NVIDIA Vision API (Llama 3.2 11B Vision / Nemotron).
    Inherits 3-tier vehicle crop & bottom ROI fallback pipeline from BasePlateRecognizer.
    """

    def __init__(self, api_key: str | None = None, invoke_url: str | None = None, model_name: str | None = None):
        self.api_key = api_key
        self.invoke_url = invoke_url or settings.NVIDIA_INVOKE_URL
        self.model_name = model_name or "meta/llama-3.2-11b-vision-instruct"

    def _get_api_keys(self) -> list[str]:
        keys = []
        if self.api_key:
            keys.append(self.api_key)
        if settings.NEMOTRON_API_KEY and settings.NEMOTRON_API_KEY not in keys:
            keys.append(settings.NEMOTRON_API_KEY)
        if settings.LLAMA_API_KEY and settings.LLAMA_API_KEY not in keys:
            keys.append(settings.LLAMA_API_KEY)
        return keys

    def _get_base64_and_mime(self, image_input: str | bytes, filename: str) -> tuple[str, str]:
        if isinstance(image_input, bytes):
            raw_data = image_input
            ext = os.path.splitext(filename)[1].lower().replace(".", "")
        else:
            with open(image_input, "rb") as img_file:
                raw_data = img_file.read()
            ext = os.path.splitext(image_input)[1].lower().replace(".", "")

        base64_str = base64.b64encode(raw_data).decode("utf-8")
        mime_type = "image/jpeg" if ext in ("jpg", "jpeg", "") else f"image/{ext}"
        return base64_str, mime_type

    def _recognize_single_image(
        self, image_input: str | bytes, filename: str = "image.jpg"
    ) -> list[dict[str, Any]]:
        """Process a single image crop or full image with NVIDIA Vision API."""
        keys_to_try = self._get_api_keys()
        if not keys_to_try:
            raise ValueError("No NVIDIA API keys (LLAMA_API_KEY / NEMOTRON_API_KEY) configured in settings/env.")

        base64_image, mime_type = self._get_base64_and_mime(image_input, filename)

        payload: dict[str, Any] = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Identify and extract the Indian vehicle license plate number from this image. Return ONLY the license plate alphanumeric string (e.g. RJ09GA0165 or MH01AB1234).",
                        },
                        {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{base64_image}"}},
                    ],
                }
            ],
            "model": self.model_name,
            "max_tokens": 128,
            "temperature": 0.1,
            "stream": False,
        }

        for key in keys_to_try:
            headers = {
                "Authorization": f"Bearer {key}",
                "Accept": "application/json",
            }
            try:
                response = requests.post(
                    self.invoke_url,
                    headers=headers,
                    json=payload,
                    timeout=(settings.HTTP_CONNECT_TIMEOUT, settings.HTTP_READ_TIMEOUT),
                )
                if response.status_code == 200:
                    try:
                        res_json = response.json()
                    except ValueError:
                        logger.error("[NvidiaVisionStrategy] 200 response was not valid JSON.")
                        continue

                    raw_text = extract_message_content(res_json)
                    if raw_text is None:
                        logger.error(
                            "[NvidiaVisionStrategy] 200 response had no usable message content; "
                            f"top-level keys: {sorted(res_json)[:8] if isinstance(res_json, dict) else type(res_json).__name__}"
                        )
                        continue

                    matches = list(INDIAN_PLATE_REGEX.finditer(raw_text))
                    output = []
                    for match in matches:
                        info = self.parse_plate_info(match.group(0))
                        if info:
                            info["raw_text"] = raw_text
                            output.append(info)

                    if output:
                        return output
                    elif raw_text:
                        return [{"plate": "N/A", "state": "N/A", "raw_text": raw_text}]
                else:
                    logger.error(
                        f"[NvidiaVisionStrategy] Error {response.status_code} with key '{key[:12]}...': {response.text}"
                    )
            except (requests.RequestException, ValueError, KeyError, TimeoutError) as e:
                logger.error(f"[NvidiaVisionStrategy] Exception with key '{key[:12]}...': {e}")

        return []
