class PromptTemplates:
    SYSTEM = """
You must return ONLY valid JSON.
No explanation text.
"""

    TEMPLATE = """
{system}

User Input:
{user_input}

Return JSON in this format:
{{
  "reply": string,
  "word_count": number
}}
"""

    @classmethod
    def build(cls, user_input: str) -> str:
        return cls.TEMPLATE.format(
            system=cls.SYSTEM.strip(),
            user_input=user_input.strip()
        )
