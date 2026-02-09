from ollama import chat
from ollama import ChatResponse

print("Ask a question to the model:")
user_input = input()

response: ChatResponse = chat(model='gemma3:270m', messages=[
  {
    'role': 'user',
    'content': user_input,
  },
])
#print(response['message']['content'])
# or access fields directly from the response object
print(response.message.content)