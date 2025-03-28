"""
Main application module.
"""
import uuid
from typing import Sequence

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    trim_messages,
)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph
from typing_extensions import Annotated, TypedDict

from src.config import Config
from src.utils.logger import get_logger

# Load environment variables from .env file
load_dotenv()

# Initialize logger
logger = get_logger("main")

human_message_hi_im_bob = HumanMessage(content="Hi! I'm Bob")
human_message_what_my_name = HumanMessage(content="What's my name?")


class State(TypedDict):
    """
    TypedDict representing the state of a conversation in the LangGraph application.
    Unlike the default MessagesState that only tracks conversation messages, this custom State adds the ability to track language preferences,
    which could be useful for multilingual chatbot applications.

    Consider that built in MessageState is defined as follows:

    class MessagesState(TypedDict):
        messages: Annotated[list[AnyMessage], add_messages]

    Attributes:
        messages: A sequence of BaseMessage objects representing the conversation history.
               The add_messages annotation enables proper message handling in LangGraph.
        language: A string indicating the language preference for the conversation.
    """

    messages: Annotated[Sequence[BaseMessage], add_messages]
    language: str


def stateless_chat(llm: BaseChatModel):
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
def call_model(state: State, llm: BaseChatModel, trimmer):
    # You need to trim messages BEFORE the prompt template but AFTER you load previous messages from Message History.
    trimmed_messages = trimmer.invoke(state["messages"])

    system_template = """
    In your response use language: {language}.

    """
    prompt_template = ChatPromptTemplate.from_messages(
        [
            SystemMessage(content=system_template),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )

    # prompt = prompt_template.invoke(state)
    logger.info(f"language: {state['language']}")
    prompt = prompt_template.invoke(
        {"messages": trimmed_messages, "language": state["language"]}
    )
    response = llm.invoke(prompt)
    return {"messages": response}


# LangGraph implements a built-in persistence layer, making it ideal for chat applications that support multiple conversational turns.
def langgraph_chat(llm: BaseChatModel, token_limit: int) -> CompiledStateGraph:
    # Wrapping our chat model in a minimal LangGraph application allows us
    # to automatically persist the message history, simplifying the development of multi-turn applications.

    # Define a new graph
    workflow = StateGraph(state_schema=State)

    trimmer = build_message_trimmer(llm, token_limit)

    # Define the (single) node in the graph
    workflow.add_edge(start_key=START, end_key="model")
    workflow.add_node(
        node="model", action=lambda state: call_model(state, llm, trimmer)
    )

    # Add memory
    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)
    return app


def stateful_chat(llm: BaseChatModel, token_limit: int):
    app = langgraph_chat(llm, token_limit)
    chat_thread_id = str(uuid.uuid4())
    # We now need to create a config that we pass into the runnable every time.
    # This config contains information that is not part of the input directly, but is still useful.
    # In this case, we want to include a thread_id.
    config = {"configurable": {"thread_id": chat_thread_id}}
    # This enables us to support multiple conversation threads with a single application, a common requirement when your application has multiple users.
    # Every conversation has an associated thread_id. So through the usage of thread_id is possible to switch between different conversations.

    output = app.invoke(
        State(messages=[human_message_hi_im_bob], language="italian"), config
    )

    # Because of the threadId is the same I don't need to specify again the language I asked for to reply
    output = app.invoke(State(messages=[human_message_what_my_name]), config)

    output = app.invoke(
        State(
            messages=[HumanMessage(content="Which language I asked you to reply with?")]
        ),
        config,
    )
    # Having set token limit for trimmer, it shouldn't remember the name provided in the first message
    for msg in output["messages"]:
        msg.pretty_print()


def build_message_trimmer(llm: BaseChatModel, token_limit: int = 100):
    # One important concept to understand when building chatbots is how to manage conversation history.
    # If left unmanaged, the list of messages will grow unbounded and potentially overflow the context window of the LLM.
    # Therefore, it is important to add a step that limits the size of the messages you are passing in.
    #
    # We can do this by adding a simple step in front of the prompt that modifies the messages key appropriately,
    # and then wrap that new chain in the Message History class.
    #
    # The trimmer allows us to specify how many tokens we want to keep.
    # Here we are using the trim_messages function to limit the number of tokens in the messages to 65.
    # About the other parameters: we want to always keep the system message and do not allow partial messages
    trimmer = trim_messages(
        max_tokens=token_limit,
        strategy="last",
        token_counter=llm,
        include_system=True,
        allow_partial=False,
        start_on="human",
    )
    return trimmer


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
    stateful_chat(llm=llm, token_limit=10000)


if __name__ == "__main__":
    main()
