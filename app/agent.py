from langchain.schema.runnable import RunnableLambda
from langchain_groq import ChatGroq
from .image_utils import preprocess_image
from .classifier import classify_image
from .prompts import summary_prompt

# Step 1: Preprocess image (resize etc.)
image_preprocessor = RunnableLambda(lambda img: preprocess_image(img))

# Step 2: Classify satellite image
classifier_chain = RunnableLambda(lambda img: classify_image(img))

# Step 3: Check if classification confidence is low
def is_uncertain(output):
    return output["confidence"] < 0.6

# Step 4: LLM summary generator
llm = ChatGroq(model="qwen-qwq-32b", temperature=0.4)

summary_chain = (
    RunnableLambda(lambda result: {"label": result["label"]}) |
    summary_prompt |
    llm
)

# Step 5: Main LCEL Vision Agent Pipeline
image_analysis_agent = (
    image_preprocessor
    | classifier_chain
    | RunnableLambda(lambda result: summary_chain.invoke(result) if not is_uncertain(result)
                     else "Model unsure — human review recommended.")
)