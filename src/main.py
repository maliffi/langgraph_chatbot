"""
Main application module.
"""
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, HumanMessage

from src.config import Config
from src.utils.logger import get_logger

# Load environment variables from .env file
load_dotenv()

# Initialize logger
logger = get_logger("main")


def stateless_chat(llm):
    # The model on its own does not have any concept of state. So if you send the following prompt and then ask: "What's my name?"
    # It probably will reply with something like "I don't know" or "I don't have that information"
    # This because it doesn't take the previous conversation turn into context.
    human_message_hi_im_bob = HumanMessage(content="Hi! I'm Bob")
    response_hi_bob = llm.invoke([human_message_hi_im_bob])
    logger.info(f"Prompt: {human_message_hi_im_bob}")
    logger.info(f"Response: {response_hi_bob.content}")

    human_message_what_my_name = HumanMessage(content="What's my name?")
    response_what_my_name = llm.invoke([human_message_what_my_name])
    logger.info(f"Prompt: {human_message_what_my_name}")
    logger.info(f"Response: {response_what_my_name.content}")

    logger.info("\n----------------------\n")
    # To get around this, we need to pass the entire conversation history into the model.
    prompts = [
        human_message_hi_im_bob,
        AIMessage(content=response_hi_bob.content),
        human_message_what_my_name,
    ]
    response_with_context = llm.invoke(prompts)
    logger.info(f"Prompts: {prompts}")
    logger.info(f"Response: {response_with_context.content}")


def main():
    """
    Main entry point for the application.
    """
    logger.info(f"Application started in {Config.APP_MODE} mode")

    # Initialize chat model
    llm = init_chat_model(model=Config.LLM, model_provider="ollama")
    stateless_chat(llm)


if __name__ == "__main__":
    main()
