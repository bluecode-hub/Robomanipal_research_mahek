from google import genai

client = genai.Client()

user_input = input("Enter text: ")

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=user_input
)

print(response.text)
