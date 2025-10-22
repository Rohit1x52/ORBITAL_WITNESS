from langchain_core.prompts import ChatPromptTemplate

# --- ADD THIS NEW PROMPT ---
SUMMARY_PROMPT = ChatPromptTemplate.from_template(
    """
    You are a satellite image analyst. Provide a brief, one-sentence summary
    for the following event: {label}
    """
)


# --- This is our existing RAG prompt ---
SOLUTION_PROMPT = ChatPromptTemplate.from_template(
# ... existing code ...
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
