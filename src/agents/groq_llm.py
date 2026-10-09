"""
Groq LLM Client & Tool Calling Adapter
Integrates Groq API (e.g. llama-3.3-70b-versatile or mixtral-8x7b-32768)
with automatic fallback to deterministic routing if no API key is provided.
"""
import os
import json
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

class GroqAgentClient:
    def __init__(self, model_name: str = "llama-3.3-70b-versatile"):
        self.api_key = os.getenv("GROQ_API_KEY", "").strip()
        self.model_name = model_name
        self.client = None
        if self.api_key:
            try:
                self.client = Groq(api_key=self.api_key)
            except Exception as e:
                print(f"Warning: Failed to initialize Groq client: {e}")

    @property
    def is_available(self) -> bool:
        return self.client is not None

    def synthesize_response(self, user_prompt: str, tool_name: str, tool_output: dict, system_context: str = "") -> str:
        """Uses Groq LLM to synthesize a natural, executive-ready marketer answer from tool outputs."""
        if not self.is_available:
            return None

        system_msg = (
            "You are an expert AI Marketing Copilot for an enterprise Customer Data Platform (CDP). "
            "You provide concise, actionable, and executive-level responses based strictly on the tool outputs provided. "
            "Never reveal raw unmasked PII. Always format metrics clearly (e.g., currency, percentages)."
            + ("\n" + system_context if system_context else "")
        )

        user_content = f"""User Request: "{user_prompt}"
Executed Tool: {tool_name}
Tool Output JSON:
{json.dumps(tool_output, indent=2, default=str)}

Synthesize a helpful, professional marketing analysis answering the user's intent. Include key numbers and recommended next steps."""

        try:
            chat_completion = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_content}
                ],
                model=self.model_name,
                temperature=0.2,
                max_tokens=600
            )
            return chat_completion.choices[0].message.content
        except Exception as e:
            print(f"Groq API call error: {e}")
            return None
