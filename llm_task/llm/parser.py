import json

class OutputValidationError(Exception):
    pass

class OutputParser:
    @staticmethod
    def parse(raw_output: str) -> dict:
        try:
            data = json.loads(raw_output)
        except json.JSONDecodeError:
            raise OutputValidationError("Invalid JSON output")

        if "reply" not in data or "word_count" not in data:
            raise OutputValidationError("Missing required fields")

        if not isinstance(data["reply"], str):
            raise OutputValidationError("reply must be a string")

        if not isinstance(data["word_count"], int):
            raise OutputValidationError("word_count must be an integer")

        return data
