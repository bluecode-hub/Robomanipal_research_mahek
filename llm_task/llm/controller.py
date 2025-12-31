from llm.prompt import PromptTemplates
from llm.parser import OutputParser, OutputValidationError
from llm.logger import logger

class LLMController:
    def __init__(self, client, max_retries=2):
        self.client = client
        self.max_retries = max_retries

    def run(self, user_input: str) -> dict:
        prompt = PromptTemplates.build(user_input)
        logger.info("Prompt used:\n%s", prompt)

        for attempt in range(1, self.max_retries + 1):
            logger.info("Attempt %d", attempt)

            raw = self.client.generate(prompt)
            logger.info("Raw response:\n%s", raw)

            try:
                return OutputParser.parse(raw)
            except OutputValidationError as e:
                logger.error("Validation failed: %s", e)

        raise RuntimeError("LLM failed after retries")
