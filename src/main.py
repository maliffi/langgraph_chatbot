"""
Main application module.
"""
import uuid

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.graph.state import CompiledStateGraph

from src.config import Config
from src.utils.logger import get_logger

# Load environment variables from .env file
load_dotenv()

# Initialize logger
logger = get_logger("main")

human_message_hi_im_bob = HumanMessage(content="Hi! I'm Bob")
human_message_what_my_name = HumanMessage(content="What's my name?")


def stateless_chat(llm):
    # The model on its own does not have any concept of state. So if you send the following prompt and then ask: "What's my name?"
    # It probably will reply with something like "I don't know" or "I don't have that information"
    # This because it doesn't take the previous conversation turn into context.
    response_hi_bob = llm.invoke([human_message_hi_im_bob])
    human_message_hi_im_bob.pretty_print()
    response_hi_bob.pretty_print()

    response_what_my_name = llm.invoke([human_message_what_my_name])
    human_message_what_my_name.pretty_print()
    response_what_my_name.pretty_print()

    logger.info("\n----------------------\n")
    # To get around this, we need to pass the entire conversation history into the model.
    prompts = [
        human_message_hi_im_bob,
        AIMessage(content=response_hi_bob.content),
        human_message_what_my_name,
    ]
    response_with_context = llm.invoke(prompts)
    for prompt in prompts:
        prompt.pretty_print()
    response_with_context.pretty_print()


# Define the function that calls the model
def call_model(state: MessagesState, llm: BaseChatModel):
    response = llm.invoke(state["messages"])
    return {"messages": response}


# LangGraph implements a built-in persistence layer, making it ideal for chat applications that support multiple conversational turns.
def langgraph_chat(llm: BaseChatModel) -> CompiledStateGraph:
    # Wrapping our chat model in a minimal LangGraph application allows us
    # to automatically persist the message history, simplifying the development of multi-turn applications.

    # Define a new graph
    workflow = StateGraph(state_schema=MessagesState)

    # Define the (single) node in the graph
    workflow.add_edge(start_key=START, end_key="model")
    workflow.add_node(node="model", action=lambda state: call_model(state, llm))

    # Add memory
    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)
    return app


def main():
    """
    Main entry point for the application.
    """
    logger.info(f"Application started in {Config.APP_MODE} mode")

    # Initialize chat model
    llm = init_chat_model(model=Config.LLM, model_provider=Config.LLM_PROVIDER)
    logger.info("\n------STATELESS CHAT-------")
    stateless_chat(llm)

    logger.info("\n------STATEFUL CHAT-------")
    app = langgraph_chat(llm)
    chat_thread_id = str(uuid.uuid4())
    # We now need to create a config that we pass into the runnable every time.
    # This config contains information that is not part of the input directly, but is still useful.
    # In this case, we want to include a thread_id.
    config = {"configurable": {"thread_id": chat_thread_id}}
    # This enables us to support multiple conversation threads with a single application, a common requirement when your application has multiple users.
    # Every conversation has an associated thread_id. So through the usage of thread_id is possible to switch between different conversations.

    input_messages = [human_message_hi_im_bob]
    output = app.invoke({"messages": input_messages}, config)
    for msg in output["messages"]:
        msg.pretty_print()


if __name__ == "__main__":
    main()
