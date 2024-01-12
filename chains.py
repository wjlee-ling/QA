from prompts import prompt_context, parser_context

import os
from langchain.chat_models import ChatOpenAI

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
model = ChatOpenAI(
    model_name="gpt-4-1106-preview",
    temperature=0.3,
    api_key=OPENAI_API_KEY,
)

chain_context = prompt_context | model | parser_context
