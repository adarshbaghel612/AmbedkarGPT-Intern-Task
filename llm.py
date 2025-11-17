from langchain_community.llms import Ollama

llm = Ollama(
    model="qwen2:0.5b",  # change to mistral, phi3, etc.
    temperature=0.2,
)

response = llm.invoke("Explain machine learning in simple words.")
print(response)