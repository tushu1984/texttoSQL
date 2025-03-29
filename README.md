# TextToSQL Chatbot

A natural language to SQL query converter application that allows users to interact with the Sakila Movie Database through plain English questions.

## Features

- **Natural Language Processing**: Ask questions in plain English and get answers from the database
- **Interactive Chat Interface**: Clean, modern UI with chat history and formatting
- **SQL Transparency**: View the actual SQL queries being executed
- **Data Visualizations**: Generate charts and graphs from query results
- **Table Formatting**: Format results in clean, readable tables

## Tech Stack

- **Backend**: Python, LangChain, OpenAI API
- **Database**: MySQL (Sakila Database)
- **Frontend**: Streamlit
- **Visualization**: Plotly Express
- **Data Processing**: Pandas

## Setup Instructions

### Prerequisites

- Python 3.8+
- MySQL Server with Sakila Database installed
- OpenAI API key

### Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/TextToSQL.git
   cd TextToSQL
   ```

2. Create and activate a virtual environment:
   ```
   python -m venv .venv
   # On Windows
   .venv\Scripts\activate
   # On macOS/Linux
   source .venv/bin/activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Create a `.env` file with the following content:
   ```
   OPENAI_API_KEY=your_openai_api_key
   MYSQL_USER=your_mysql_username
   MYSQL_PASSWORD=your_mysql_password
   MYSQL_HOST=localhost
   MYSQL_DATABASE=sakila
   ```

5. Set up the Sakila database (if not already installed):
   ```
   python setup_sakila.py
   ```

6. Run the application:
   ```
   python main.py
   ```

   Or directly with Streamlit:
   ```
   streamlit run app.py
   ```

## Usage

1. Start the application using one of the methods above
2. Ask questions in natural language about the Sakila movie database
3. View the friendly responses and explore the data
4. Request visualizations by including terms like "chart", "graph", or "visualization" in your questions

## Example Questions

- Which movies are the most rented?
- How many customers have rented movies from New York?
- Show me a pie chart of films by rating
- What's the revenue breakdown by film category?
- Display the customer distribution by country

## Project Structure

- `app.py`: Main Streamlit application
- `app_fixed.py`: Fixed version of the app with all features
- `main.py`: Entry point for the application
- `requirements.txt`: Project dependencies
- `check_sakila.py`: Script to check if Sakila DB is available
- `setup_sakila.py`: Script to set up Sakila database
- `run.bat`: Batch file to run the application on Windows

## License

MIT

## Acknowledgements

- [Sakila Sample Database](https://dev.mysql.com/doc/sakila/en/)
- [LangChain](https://python.langchain.com/)
- [OpenAI](https://openai.com/)
- [Streamlit](https://streamlit.io/) 