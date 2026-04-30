from model import retriever, llm, prompt
query = "what is the cfastest goal in history of UCL"
docs = retriever.invoke(query)

context = "\n\n".join(doc.page_content for doc in docs)

try:
    formatted = prompt.format(context=context, question=query)
    result = llm.invoke(formatted).content
except Exception as e:
    print("Error occurred:", e)

print("Query:", query)
print("Context:", context)
print("Answer:", result)