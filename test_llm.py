from llm_engine import LocalLLM

print("Loading LLM...")

llm = LocalLLM()

response = llm.generate("What is Power BI?")

print("\nResponse:\n")
print(response)