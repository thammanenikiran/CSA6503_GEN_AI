from google import genai

client = genai.Client(api_key="AQ.Ab8RN6IbHqq0yOQsyj3Upr6SfORSWlxvRfSvJRvUxZ4qOVCqow")

response = client.models.generate_content(
    model="gemini-3.5-flash",
    contents="What is AI?"
)

print(response.text)