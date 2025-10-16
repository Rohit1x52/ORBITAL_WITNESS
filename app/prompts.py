# app/prompts.py
from langchain_core.prompts import ChatPromptTemplate

# We no longer need the old CLASSIFICATION_PROMPT

SOLUTION_PROMPT = ChatPromptTemplate.from_template(
    """
    You are an expert disaster management and urban planning advisor.
    Your task is to provide solutions based on a detected event and relevant context.

    **Detected Event:** {event_class}
    **Summary of Event:** {summary}

    **Retrieved Context from Knowledge Base:**
    ---
    {context}
    ---

    Based on all the information above, provide a structured response with:
    1.  A concise **Short-Term Solution**.
    2.  A strategic **Long-Term Solution**.
    """
)