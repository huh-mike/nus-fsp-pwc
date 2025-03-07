# NUS-FSP-PWC Project Guide

## Commands
- **Run application**: `python main.py`
- **Install dependencies**: `pip install -r requirements.txt`
- **Setup environment**: `python -m venv venv && source venv/bin/activate` (macOS/Linux) or `python -m venv venv && venv\Scripts\activate` (Windows)

## Code Style Guidelines
- **Imports**: Group imports by standard library, third-party, and internal modules with a blank line between groups
- **Typing**: Use type hints for function parameters and return values
- **Documentation**: Docstrings for functions using triple quotes with parameter descriptions
- **Error Handling**: Use try/except blocks with specific exceptions and error messages
- **Naming**: 
  - Classes: PascalCase (e.g., `ChatHistory`)
  - Functions/Variables: snake_case (e.g., `update_callback`)
  - Constants: UPPERCASE (e.g., `OPENAI_PROMPT`)
- **Formatting**: Follow PEP 8 guidelines for line length (79-88 chars) and indentation (4 spaces)

## Project Structure
- Web scraping components (`DataScraperNew.py`)
- Data tagging and processing (`DataTaggerNew.py`, `CsvProcessor.py`)
- Database operations (`db.py`)
- Chat interface (`ChatBotGUI.py`)
- OpenAI API integration (`GPTServices.py`)