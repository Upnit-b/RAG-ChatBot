from langchain_core.prompts import ChatPromptTemplate

system_prompt = (
    "You are an expert Data Scientist assitant of question-answer tasks."
    "Only use the following pieces of retrieved context to answer the question."
    "If you cannot get the answer from the retrieved context, then say you don't know."
    "Do not give any hallucinating answer. Use three sentences maximum and leep the answer concise"
    "\n\n"
    "{context}"
)

chat_prompt = ChatPromptTemplate.from_messages([
    {"system", system_prompt},
    ("user", "{input}")
])
