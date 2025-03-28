# LangGraph Chatbot

This project demonstrates how to build a stateful chatbot using LangGraph and LangChain. It showcases the differences between stateless and stateful chat interactions, highlighting the importance of maintaining conversation context.

## Project Overview

This application uses LangGraph and LangChain with local LLMs through Ollama to create:

1. **Stateless Chat**: Demonstrates the limitations of a chat without persistence
2. **Stateful Chat**: Shows how LangGraph manages conversation state to maintain context
3. **Multi-user Support**: Implements thread_id based conversations for multi-user applications

The chatbot leverages LangGraph's built-in persistence layer with MemorySaver to simplify the development of multi-turn conversations.

## Project Structure

```
langgraph_chatbot/
├── logs/               # Log files directory
├── src/                # Source code
│   ├── __init__.py
│   ├── main.py         # Application entry point
│   ├── config.py       # Configuration settings
│   └── utils/          # Utility modules
│       └── logger.py   # Logging configuration
├── .env                # Environment variables
├── .env.example        # Environment variables template
├── .gitignore          # Git ignore file
├── .pre-commit-config.yaml  # Pre-commit hooks configuration
├── README.md           # Project documentation
└── requirements.txt    # Project dependencies
```

## Dependencies

This project relies on the following key dependencies:

### Core
- **langchain**: Base framework for building language model applications
- **langchain-core**: Core LangChain components
- **langgraph**: Graph-based framework for stateful LLM applications
- **langchain_community**: Community extensions for LangChain

### Infrastructure
- **python-dotenv**: Environment variable management
- **loguru**: Advanced logging

### Development
- **pytest**: Testing framework
- **black**: Code formatter
- **isort**: Import formatter
- **pre-commit**: Git hook manager

## Setup and Running Locally

### Prerequisites

- Python 3.9+
- Docker and Docker Compose (optional)
- Ollama installed locally

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/maliffi/langgraph_chatbot.git
   cd langgraph_chatbot
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables by creating/modifying the `.env` file (see `.env.example` for a template).

### Start Ollama (for LLM)

Make sure Ollama is running with the LLM specified in your `.env` file:

```bash
ollama pull llama2  # Or whatever model you want to use
ollama serve  # Start the Ollama server
```

### Running the Application

To run the application, use the following command:

```bash
python -m src.main
```

The application will:
1. Load the specified LLM via Ollama
2. Demonstrate a stateless chat interaction
3. Demonstrate a stateful chat with memory persistence
4. Log the results

#### Example Output
Here's an example of the output you should see:

```shell
2025-03-28 15:59:38 | INFO     | __main__:main:76 - Application started in development mode
2025-03-28 15:59:38 | INFO     | __main__:main:80 -
------STATELESS CHAT-------
================================ Human Message =================================

Hi! I'm Bob
================================== Ai Message ==================================

Hi Bob! It's nice to meet you. Is there something I can help you with or would you like to chat?
================================ Human Message =================================

What's my name?
================================== Ai Message ==================================

I don't have any information about you, so I'm not sure what your name is. We just started our conversation, and I don't have any prior knowledge about you. Would you like to tell me your name?
2025-03-28 15:59:40 | INFO     | __main__:stateless_chat:37 -
----------------------

================================ Human Message =================================

Hi! I'm Bob
================================== Ai Message ==================================

Hi Bob! It's nice to meet you. Is there something I can help you with or would you like to chat?
================================ Human Message =================================

What's my name?
================================== Ai Message ==================================

You told me your name earlier - Bob! You said "Hi, I'm Bob".
2025-03-28 15:59:40 | INFO     | __main__:main:83 -
------STATEFUL CHAT-------
================================ Human Message =================================

Hi! I'm Bob
================================== Ai Message ==================================
```

In this example, you can see the difference between:
- **Stateless Chat**: When asked "What's my name?", the model initially doesn't remember the previous context
- **Stateful Chat with Context**: When the entire conversation history is passed to the model, it can correctly identify the name
- **LangGraph Chat**: Using LangGraph's persistence layer, the model automatically maintains conversation context

## Development

This project uses several development tools to maintain code quality:

- **Black**: Code formatter that enforces a consistent style
- **isort**: Import statement formatter
- **pre-commit**: Git hook manager that runs checks before commits

To set up the pre-commit hooks:
```bash
pre-commit install
```

## Testing

Run the tests using pytest:
```bash
pytest
```

## Logging

The application uses Loguru for logging. Logs are stored in the `logs/` directory.

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request
