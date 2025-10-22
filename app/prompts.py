from langchain.prompts import PromptTemplate

summary_prompt = PromptTemplate.from_template(
    """
    You are a satellite imagery expert. Based on the detected class: '{label}',
    write a short but clear explanation of what is observed in the satellite image.
    Use technical terms when necessary and explain the possible cause and impact.
    After that list down atleast 5 temporary and 5 permanent solutions for that in bulletpoints.
    """
)