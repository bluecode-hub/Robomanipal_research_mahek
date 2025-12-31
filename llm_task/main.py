from llm.client import GeminiClient
from llm.controller import LLMController

client = GeminiClient()
controller = LLMController(client)

user_input = input("Enter text: ")

result = controller.run(user_input)
print(result)
