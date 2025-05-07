# Defining the system prompt template used by the language model

# This prompt instructs the LLM to behave like a helpful assistant.
# It uses the retrieved context (passed as {context}) to answer the user's question.
# The answer should be brief (max 3 sentences), and if the answer isn't known, the assistant should say so.

system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}"
)
