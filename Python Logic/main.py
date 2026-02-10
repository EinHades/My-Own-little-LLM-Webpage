from ollama import chat
from ollama import ChatResponse

print("Ask a question to the model:")
user_input = input()
user_model_input = input("Enter the model name:")

response: ChatResponse = chat(model= user_model_input, messages=[
  {
    'role': 'user',
    'content': user_input,
  },
])
#print(response['message']['content'])
# or access fields directly from the response object
print(response.message.content)