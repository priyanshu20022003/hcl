"""
Sarvam AI client wrapper.

Provides chat completions, translation, TTS, STT, and document-intelligence
via the official sarvamai SDK.
"""

import base64
import time
from pathlib import Path

import requests


class SarvamClient:
    """Thin wrapper around the Sarvam AI Python SDK."""

    def __init__(self, api_key: str, chat_model: str = "sarvam-m"):
        self.api_key = api_key
        self.chat_model = chat_model
        self.enabled = bool(api_key)
        self.client = None

        if self.enabled:
            try:
                from sarvamai import SarvamAI
                self.client = SarvamAI(api_subscription_key=api_key)
            except Exception as exc:
                print(f"[sarvam] Failed to initialise SDK: {exc}")
                self.enabled = False
                self.client = None

    # ------------------------------------------------------------------
    # Chat completions
    # ------------------------------------------------------------------
    def chat(self, system_prompt: str, user_prompt: str) -> str | None:
        """Send a chat completion request and return the assistant reply."""
        if not self.enabled or not self.client:
            return None
        try:
            response = self.client.chat.completions(
                model=self.chat_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            return response.choices[0].message.content
        except Exception as exc:
            print(f"[sarvam] chat error: {exc}")
            return None

    # ------------------------------------------------------------------
    # Translation
    # ------------------------------------------------------------------
    def translate(self, text: str, source_lang: str, target_lang: str) -> str | None:
        """Translate text between languages."""
        if not self.enabled or not self.client:
            return None
        try:
            response = self.client.text.translate(
                input=text,
                source_language_code=source_lang,
                target_language_code=target_lang,
            )
            return (
                getattr(response, "translated_text", None)
                or getattr(response, "translation", None)
                or getattr(response, "text", None)
            )
        except Exception as exc:
            print(f"[sarvam] translate error: {exc}")
            return None

    # ------------------------------------------------------------------
    # Speech-to-text
    # ------------------------------------------------------------------
    def speech_to_text(self, file_path: Path) -> dict | None:
        """Transcribe an audio file."""
        if not self.enabled or not self.client:
            return None
        try:
            with open(file_path, "rb") as audio_file:
                response = self.client.speech_to_text.transcribe(file=audio_file)
            return {
                "transcript": getattr(response, "transcript", ""),
                "language_code": getattr(response, "language_code", "unknown"),
            }
        except Exception as exc:
            print(f"[sarvam] STT error: {exc}")
            return None

    # ------------------------------------------------------------------
    # Text-to-speech
    # ------------------------------------------------------------------
    def text_to_speech(self, text: str, target_language: str) -> str | None:
        """Convert text to speech and return base64-encoded audio."""
        if not self.enabled or not self.client:
            return None
        try:
            response = self.client.text_to_speech.convert(
                text=text,
                target_language_code=target_language,
            )
            audio_bytes = getattr(response, "audio", None) or getattr(response, "audio_bytes", None)
            if not audio_bytes:
                return None
            if isinstance(audio_bytes, str):
                return audio_bytes
            return base64.b64encode(audio_bytes).decode("utf-8")
        except Exception as exc:
            print(f"[sarvam] TTS error: {exc}")
            return None

    # ------------------------------------------------------------------
    # Document intelligence (OCR)
    # ------------------------------------------------------------------
    def extract_document_text(self, file_path: Path, language: str = "en-IN") -> str | None:
        """Extract text from a document using Sarvam document intelligence.
        
        This handles the asynchronous job creation, polling, and content fetching.
        """
        if not self.enabled or not self.client:
            return None
            
        try:
            # 1. Create the job
            print(f"[sarvam] Creating doc-intel job for {file_path.name}...")
            job = self.client.document_intelligence.create_job(
                language=language,
                output_format="md",
            )
            
            # 2. Upload the file
            job.upload_file(str(file_path))
            
            # 3. Poll for completion
            max_retries = 30  # ~30-60 seconds max
            retry_count = 0
            status = "PENDING"
            while retry_count < max_retries:
                status = job.get_status()
                print(f"[sarvam] Job {job.job_id} status: {status}")
                
                if status == "COMPLETED":
                    break
                elif status in ("FAILED", "CANCELLED"):
                    print(f"[sarvam] Job failed with status: {status}")
                    return None
                    
                time.sleep(2)
                retry_count += 1
            
            if status != "COMPLETED":
                print("[sarvam] Job timed out.")
                return None
            
            # 4. Get download URLs and fetch the markdown content
            urls = job.get_download_urls()
            if not urls:
                print("[sarvam] No download URLs returned.")
                return None
            
            # Usually returns a list of signed URLs. We want the markdown one.
            md_url = None
            if isinstance(urls, dict):
                md_url = urls.get("md") or urls.get("markdown")
            elif isinstance(urls, list) and urls:
                for u in urls:
                    if ".md" in u.lower() or "markdown" in u.lower():
                        md_url = u
                        break
                if not md_url:
                    md_url = urls[0]
            
            if not md_url:
                print("[sarvam] Could not find markdown URL.")
                return None
                
            # 5. Fetch the actual content
            resp = requests.get(md_url, timeout=20)
            resp.raise_for_status()
            return resp.text
            
        except Exception as exc:
            print(f"[sarvam] doc-extract error: {exc}")
            return None
