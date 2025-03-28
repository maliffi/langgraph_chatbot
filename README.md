# LangChain Text Classification

This project demonstrates how to use LangChain to create a text classification system and has been developed starting from the LangChain Text Classification tutorial available [here](https://python.langchain.com/docs/tutorials/classification/).
It leverages LangChain's structured output capabilities with Ollama models to classify text based on sentiment, aggressiveness level, and language identification.

## Project Overview

This application uses LangChain and local LLMs through Ollama to perform multi-aspect text classification:

1. **Sentiment Analysis**: Determines if text is positive, negative, or neutral
2. **Aggressiveness Scoring**: Rates content on a scale from 1-10 for aggressiveness
3. **Language Detection**: Identifies the language of the text

The classification is implemented using LangChain's structured output capability with Pydantic models to ensure consistent and typed responses from language models.

## Project Structure

```
langchain_classification/
├── logs/               # Log files directory
├── src/                # Source code
│   ├── __init__.py
│   ├── main.py         # Application entry point
│   ├── config.py       # Configuration settings
│   ├── models/         # Pydantic model definitions
│   │   └── classification.py  # Classification schema
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
- **langchain-ollama**: Integration with Ollama for local LLMs
- **pydantic**: Data validation and settings management

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
   git clone https://github.com/maliffi/langchain_classification.git
   cd langchain_classification
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
2. Process an example text
3. Classify the text based on sentiment, aggressiveness, and language
4. Log the results

#### Example Output
Here's an example of the output you should see:

```
2025-03-28 12:03:58 | INFO     | __main__:main:22 - Application started in development mode
2025-03-28 12:03:59 | INFO     | __main__:main:42 - Input to classify: Sono incredibilmente contento di averti conosciuto! Sono sicuro diventeremo buoni amici!
2025-03-28 12:03:59 | INFO     | __main__:main:43 - Response: Sentiment: positive, Aggressiveness: 1, Language: Italian
```

In this example, the Italian text "Sono incredibilmente contento di averti conosciuto! Sono sicuro diventeremo buoni amici!" (which means "I'm incredibly happy to have met you! I'm sure we'll become good friends!") is classified as:

- **Sentiment**: positive
- **Aggressiveness**: 1 (very low)
- **Language**: Italian

You can modify the input text in `src/main.py` to test different classifications.

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
