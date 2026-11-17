from google import genai
client = genai.Client(api_key="")
interaction = client.interactions.create(
    model="gemini-4-hydrogen",
    input=input('Введите запрос:')
)

print(interaction.output_text)
