import os
import streamlit as st
from dotenv import load_dotenv
from langchain.llms import OpenAI
from langchain_openai import ChatOpenAI
from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.sql_database import SQLDatabase
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage, AIMessage
import mysql.connector
from mysql.connector import Error
import re
from tabulate import tabulate
import urllib.parse
import pandas as pd
import plotly.express as px
import json

# Load environment variables
load_dotenv()

# Set page configuration
st.set_page_config(
    page_title="Text to SQL Chatbot", 
    page_icon="💬", 
    layout="wide"
)

# Custom CSS styling
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .css-1d391kg, .css-12oz5g7 {
        padding: 2rem 1rem;
    }
    .stButton>button {
        background-color: #4361ee;
        color: white;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #3a56d4;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .stTextInput>div>div>input {
        border-radius: 8px;
        border: 1px solid #e0e0e0;
        padding: 0.75rem;
    }
    .chat-message {
        padding: 1.5rem;
        border-radius: 12px;
        margin-bottom: 1.25rem;
        display: flex;
        flex-direction: column;
        box-shadow: 0 3px 10px rgba(0, 0, 0, 0.08);
    }
    .chat-message.user {
        background-color: #e7f0ff;
        border-left: 4px solid #4361ee;
    }
    .chat-message.assistant {
        background-color: white;
        border-left: 4px solid #4cc9f0;
    }
    h1 {
        color: #1e3a8a;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    .sql-code {
        background-color: #272822;
        color: #f8f8f2;
        border-radius: 8px;
        padding: 1rem;
        font-family: 'Courier New', monospace;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        white-space: pre-wrap;
        overflow-x: auto;
    }
    .stAlert {
        border-radius: 8px;
    }
    .annotation-container {
        margin-top: 1rem;
        padding: 1rem;
        background-color: #e7f0ff;
        border-radius: 8px;
        border-left: 4px solid #4361ee;
    }
    .result-table {
        width: 100%;
        border-collapse: collapse;
        margin: 1rem 0;
        font-size: 0.9em;
        font-family: 'Inter', sans-serif;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
        border-radius: 8px;
        overflow: hidden;
    }
    .result-table thead tr {
        background-color: #4361ee;
        color: #ffffff;
        text-align: left;
        font-weight: 600;
    }
    .result-table th,
    .result-table td {
        padding: 14px 16px;
    }
    .result-table tbody tr {
        border-bottom: 1px solid #f1f1f1;
    }
    .result-table tbody tr:nth-of-type(even) {
        background-color: #f9f9f9;
    }
    .result-table tbody tr:last-of-type {
        border-bottom: 2px solid #4361ee;
    }
    .explanation {
        background-color: #e7f0ff;
        padding: 1.25rem;
        border-radius: 8px;
        margin-top: 1rem;
        border-left: 4px solid #4361ee;
    }
    pre {
        white-space: pre-wrap;
    }
    .sql-section {
        margin-top: 1.5rem;
        background-color: #f1f5f9;
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
        border: 1px solid #e2e8f0;
    }
    .sql-header {
        background-color: #3b82f6;
        color: white;
        padding: 0.9rem 1.2rem;
        font-weight: 600;
        display: flex;
        align-items: center;
        gap: 0.6rem;
        font-size: 1rem;
        letter-spacing: 0.02em;
    }
    .sql-header i {
        font-size: 1.25rem;
    }
    .sql-code {
        background-color: #1e293b;
        color: #e2e8f0;
        border-radius: 0 0 8px 8px;
        padding: 1.2rem;
        font-family: 'Courier New', monospace;
        margin: 0;
        white-space: pre-wrap;
        overflow-x: auto;
        font-size: 0.95rem;
        line-height: 1.5;
    }
    .answer-highlight {
        font-size: 1.15em;
        margin-bottom: 1.5rem;
        padding: 1.5rem;
        background-color: #e6f4ff;
        border-radius: 12px;
        border-left: 5px solid #4361ee;
        line-height: 1.6;
        box-shadow: 0 3px 10px rgba(0, 0, 0, 0.08);
        color: #2c3e50;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }
    .sidebar-card {
        background-color: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
        margin-bottom: 1.5rem;
    }
    .sidebar-card h3 {
        color: #4361ee;
        font-weight: 600;
        margin-bottom: 1rem;
        font-size: 1.1rem;
    }
    .sidebar-card ul {
        padding-left: 1.25rem;
    }
    .sidebar-card li {
        margin-bottom: 0.5rem;
        line-height: 1.4;
    }
    .sidebar-card li:hover {
        color: #4361ee;
        cursor: pointer;
    }
    .schema-container {
        background-color: white;
        padding: 1rem;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
    }
    /* Mobile responsiveness */
    @media (max-width: 768px) {
        .stApp {
            padding: 0.5rem;
        }
        .chat-message {
            padding: 1rem;
        }
    }
</style>
""", unsafe_allow_html=True)

# Header
st.title("📊 Text to SQL Chatbot")
st.markdown("Ask questions about the Sakila Movie Database in natural language")

# MySQL Connection
@st.cache_resource
def get_db_connection():
    try:
        # Get credentials from environment
        mysql_user = os.getenv('MYSQL_USER')
        mysql_password = os.getenv('MYSQL_PASSWORD')
        mysql_host = os.getenv('MYSQL_HOST')
        mysql_database = os.getenv('MYSQL_DATABASE')
        
        # URL encode the password to handle special characters
        encoded_password = urllib.parse.quote_plus(mysql_password)
        
        # Create the connection string
        db_uri = f"mysql+mysqlconnector://{mysql_user}:{encoded_password}@{mysql_host}/{mysql_database}"
        db = SQLDatabase.from_uri(db_uri)
        return db
    except Error as e:
        st.error(f"Error connecting to MySQL: {e}")
        return None

# LangChain SQL Agent setup
@st.cache_resource
def create_agent(_db):
    # Set up LLM
    llm = ChatOpenAI(
        temperature=0, 
        model="gpt-3.5-turbo-16k",
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )
    
    # Create toolkit
    toolkit = SQLDatabaseToolkit(db=_db, llm=llm)
    
    # Create agent
    agent_executor = create_sql_agent(
        llm=llm,
        toolkit=toolkit,
        verbose=True,
        agent_type="openai-tools",
        handle_parsing_errors=True,
    )
    
    return agent_executor

# Check if the user wants a visualization
def detect_visualization_request(query):
    viz_keywords = {
        'chart': 'auto',
        'graph': 'auto',
        'plot': 'auto',
        'visualization': 'auto',
        'visualize': 'auto',
        'visualise': 'auto',
        'bar chart': 'bar',
        'bar graph': 'bar',
        'histogram': 'bar',
        'pie chart': 'pie',
        'pie graph': 'pie',
        'line chart': 'line',
        'line graph': 'line',
        'time series': 'line',
        'trend': 'line'
    }
    
    query_lower = query.lower()
    for keyword, viz_type in viz_keywords.items():
        if keyword in query_lower:
            return True, viz_type
    
    return False, 'auto'

# Process with LangChain agent
def process_query(agent, user_input):
    # Check if the user wants tabular format or visualization
    wants_table = any(phrase in user_input.lower() for phrase in [
        "tabular format", "table format", "in a table", "as a table", "show in table", 
        "table view", "show me a table", "tabular view", "display as table"
    ])
    
    wants_viz, viz_type = detect_visualization_request(user_input)
    
    # Modify user query if specific formats are requested
    modified_input = user_input
    if wants_table:
        table_instruction = "Please provide the answer in a well-formatted table with clear columns and values."
        if table_instruction not in modified_input:
            modified_input = modified_input + " " + table_instruction
    
    if wants_viz:
        viz_instruction = f"Please provide the data as a JSON array with properly labeled fields so I can visualize it."
        if viz_instruction not in modified_input:
            modified_input = modified_input + " " + viz_instruction
    
    prompt_template = f"""
    You are a helpful and friendly customer service representative named MovieBot for a movie rental store. 

    Answer the following question about our movie rental database in plain, simple language:
    "{modified_input}"

    IMPORTANT GUIDELINES:
    1. You must write your answer as if you are talking to a regular customer with NO technical knowledge
    2. Use a warm, friendly, conversational tone - like a helpful store employee would use
    3. Keep your main answer short and to the point (1-3 sentences)
    4. Include specific data in your answer (numbers, titles, names) but present it conversationally
    5. NEVER use technical terms like database, query, SQL, table, schema, join, etc. in your main answer
    6. Do not explain how you found the information or mention any technical processes
    7. Avoid phrases like "according to the database" or "the data shows" - just give the information directly
    8. Structure your response in two clearly separated parts:
       - FIRST PART: A friendly, conversational answer for the customer
       - SECOND PART: The SQL query in a code block with ```sql and ``` tags (this will be shown in a technical section)
    
    YOU MUST ALWAYS include the exact SQL query you executed in a code block formatted like this:
    ```sql
    SELECT * FROM table WHERE condition;
    ```
    
    This SQL part will be hidden from the customer but is required for our technical staff.
    
    Example bad answer: "I queried the database and found that according to our rental records, the movie 'BUCKET BROTHERHOOD' has been rented 34 times based on the SQL query results."
    
    Example good answer: "Our most popular movie is 'BUCKET BROTHERHOOD' - it's been rented 34 times! Customers really seem to love that one.
    
    ```sql
    SELECT film.title, COUNT(rental.rental_id) AS rental_count 
    FROM film 
    JOIN inventory ON film.film_id = inventory.film_id 
    JOIN rental ON inventory.inventory_id = rental.inventory_id 
    GROUP BY film.title 
    ORDER BY rental_count DESC 
    LIMIT 1;
    ```"
    
    Remember to separate your SQL code completely from your friendly response.
    """
    
    if wants_table:
        prompt_template += """
        9. For the table data, format it using markdown table syntax, but make sure to introduce it with a friendly sentence first:
        
        | Column1 | Column2 | Column3 |
        |---------|---------|---------|
        | Value1 | Value2 | Value3 |
        | Value4 | Value5 | Value6 |
        
        Make sure the table is properly formatted with the separator line (|---|---|) and consistent column counts.
        """
    
    if wants_viz:
        prompt_template += f"""
        9. For visualization data, include a properly formatted JSON array in a separate section of your response:
        
        ```json
        [
          {{"label": "Category1", "value": 42}},
          {{"label": "Category2", "value": 28}},
          {{"label": "Category3", "value": 15}}
        ]
        ```
        
        The JSON MUST be an array of objects with consistent keys. Use "label" for category names and "value" for numeric values when appropriate.
        For time-based visualizations, include a "date" or "time" field.
        
        This JSON will not be shown to the customer - it's only used to create the visualization.
        """
    
    prompt_template += """
    FINAL REMINDERS:
    1. Write as if you're having a friendly conversation with a customer at the rental counter
    2. Keep your answers simple, helpful, and focused on what they asked
    3. Completely separate your SQL code from your friendly response
    4. Imagine explaining movie information to someone who just wants a simple answer
    5. ALWAYS include the complete SQL query you used in a code block with proper formatting
    """
    
    response = agent.invoke({"input": prompt_template})
    return response['output'], wants_table, wants_viz, viz_type

# Extract JSON data for visualization
def extract_json_data(result):
    json_match = re.search(r'```json\s*(.*?)\s*```', result, re.DOTALL)
    if json_match:
        json_str = json_match.group(1)
        try:
            data = json.loads(json_str)
            return data, json_str
        except json.JSONDecodeError:
            return None, json_str
    return None, None

# Create visualization based on data and type
def create_visualization(data, viz_type):
    if not data or not isinstance(data, list) or len(data) == 0:
        return None
    
    try:
        df = pd.DataFrame(data)
        
        # Determine the right visualization based on data and requested type
        if viz_type == 'auto':
            # Check data structure to suggest visualization
            if len(df.columns) >= 2:
                numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
                if len(numeric_cols) >= 1:
                    if 'date' in df.columns or 'time' in df.columns or any('date' in col.lower() for col in df.columns):
                        viz_type = 'line'
                    else:
                        viz_type = 'bar'
                else:
                    viz_type = 'pie'
            else:
                viz_type = 'bar'
        
        # Create the appropriate chart
        if viz_type == 'bar':
            # Identify label and value columns
            if 'label' in df.columns and 'value' in df.columns:
                label_col, value_col = 'label', 'value'
            else:
                # Try to determine automatically
                numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
                string_cols = df.select_dtypes(include=['object']).columns
                
                if len(numeric_cols) > 0 and len(string_cols) > 0:
                    value_col = numeric_cols[0]
                    label_col = string_cols[0]
                else:
                    # Fall back to first two columns
                    label_col, value_col = df.columns[0], df.columns[1]
            
            # Sort by value for better visualization
            df = df.sort_values(by=value_col, ascending=False)
            
            fig = px.bar(
                df, 
                x=label_col, 
                y=value_col,
                labels={label_col: label_col.replace('_', ' ').title(), value_col: value_col.replace('_', ' ').title()},
                title=f"Bar Chart of {label_col.replace('_', ' ').title()} by {value_col.replace('_', ' ').title()}"
            )
            
        elif viz_type == 'pie':
            # Identify label and value columns
            if 'label' in df.columns and 'value' in df.columns:
                label_col, value_col = 'label', 'value'
            else:
                # Try to determine automatically
                numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
                string_cols = df.select_dtypes(include=['object']).columns
                
                if len(numeric_cols) > 0 and len(string_cols) > 0:
                    value_col = numeric_cols[0]
                    label_col = string_cols[0]
                else:
                    # Fall back to first two columns
                    label_col, value_col = df.columns[0], df.columns[1]
            
            fig = px.pie(
                df, 
                names=label_col, 
                values=value_col,
                title=f"Distribution of {value_col.replace('_', ' ').title()} by {label_col.replace('_', ' ').title()}"
            )
            
        elif viz_type == 'line':
            # Identify date and value columns
            date_col = next((col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()), None)
            
            if date_col:
                numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
                if len(numeric_cols) > 0:
                    value_col = numeric_cols[0]
                    
                    # Sort by date for proper line chart
                    try:
                        df[date_col] = pd.to_datetime(df[date_col])
                        df = df.sort_values(by=date_col)
                    except:
                        pass
                        
                    fig = px.line(
                        df, 
                        x=date_col, 
                        y=value_col,
                        labels={date_col: date_col.replace('_', ' ').title(), value_col: value_col.replace('_', ' ').title()},
                        title=f"Trend of {value_col.replace('_', ' ').title()} Over Time"
                    )
                else:
                    # Fallback to bar chart if no numeric columns
                    viz_type = 'bar'
                    fig = px.bar(df, x=df.columns[0], y=df.columns[1])
            else:
                # If no date column, try first two columns
                fig = px.line(
                    df, 
                    x=df.columns[0], 
                    y=df.columns[1]
                )
        else:
            # Default to bar chart for unknown types
            fig = px.bar(df, x=df.columns[0], y=df.columns[1])
        
        # Improve layout aesthetics
        fig.update_layout(
            template="plotly_white",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=20, r=20, t=40, b=20),
            font=dict(family="Arial, sans-serif", size=14),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        return fig
        
    except Exception as e:
        print(f"Visualization error: {str(e)}")
        return None

# Initialize database connection
db = get_db_connection()

if not db:
    st.error("Failed to connect to the database. Please check your database credentials.")
else:
    st.success("Successfully connected to Sakila database!")
    
    # Create SQL agent
    agent = create_agent(db)
    
    # Initialize session state for chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for message in st.session_state.messages:
        if isinstance(message, HumanMessage):
            with st.chat_message("user"):
                st.markdown(f"<div class='chat-message user'>{message.content}</div>", unsafe_allow_html=True)
        else:
            with st.chat_message("assistant"):
                st.markdown(f"<div class='chat-message assistant'>{message.content}</div>", unsafe_allow_html=True)
    
    # User input
    user_input = st.chat_input("Ask a question about the Sakila database...")
    
    if user_input:
        with st.chat_message("user"):
            st.markdown(f"<div class='chat-message user'>{user_input}</div>", unsafe_allow_html=True)
        
        # Add user message to history
        st.session_state.messages.append(HumanMessage(content=user_input))
        
        # Process with LangChain agent
        with st.spinner("Thinking..."):
            response, wants_table, wants_viz, viz_type = process_query(agent, user_input)
            
            # Extract JSON data for visualization if requested
            json_data = None
            if wants_viz:
                json_data, json_str = extract_json_data(response)
                if json_str:
                    # Remove JSON block from the response
                    response = re.sub(r'```json\s*.*?\s*```', '', response, flags=re.DOTALL | re.IGNORECASE)
            
            # Extract SQL query to display
            sql_match = re.search(r'```sql\n(.*?)\n```', response, re.DOTALL)
            sql_query = ""
            if sql_match:
                sql_query = sql_match.group(1).strip()
                # Remove SQL block from main response but will display it later
                response = re.sub(r'```sql\n.*?\n```', '', response, flags=re.DOTALL)
            else:
                # Try alternative SQL patterns that might appear in the response
                alt_patterns = [
                    r'```\n(SELECT.*?)\n```',
                    r'```\n(SELECT.*?)```',
                    r'SQL:\s*(SELECT.*?)\n\n',
                    r'query:\s*(SELECT.*?)\n\n',
                    r'(SELECT\s+.*?FROM.*?(WHERE|GROUP BY|ORDER BY|LIMIT|$).*?)(?:\n\n|\Z)'
                ]
                
                for pattern in alt_patterns:
                    alt_match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
                    if alt_match:
                        sql_query = alt_match.group(1).strip()
                        # Remove the matched SQL from the response
                        response = re.sub(pattern, '', response, flags=re.DOTALL | re.IGNORECASE)
                        break
            
            # Format tables better if they exist
            # Look for markdown tables
            table_pattern = r'\|\s*(.*?)\s*\|\s*\n\|\s*[-:\s]*\|\s*\n((?:\|\s*.*?\s*\|\s*\n)*)'
            table_matches = re.findall(table_pattern, response, re.DOTALL)
            
            for match in table_matches:
                headers = match[0]
                rows = match[1]
                
                # Extract headers
                header_cells = re.findall(r'\s*(.*?)\s*\|', '|' + headers)
                
                # Extract rows
                row_data = []
                for row in rows.strip().split('\n'):
                    cells = re.findall(r'\s*(.*?)\s*\|', '|' + row)
                    if cells:
                        row_data.append(cells)
            
                # Create HTML table - only create if we have valid header cells and rows
                if header_cells and row_data:
                    html_table = '<div class="result-table-container"><table class="result-table"><thead><tr>'
                    for header in header_cells:
                        html_table += f'<th>{header}</th>'
                    html_table += '</tr></thead><tbody>'
                    
                    for row in row_data:
                        html_table += '<tr>'
                        for cell in row:
                            html_table += f'<td>{cell}</td>'
                        html_table += '</tr>'
                    
                    html_table += '</tbody></table></div>'
                    
                    # Replace the markdown table with HTML table
                    md_table = f'|{headers}|\n|{"-" * len(headers)}|\n{rows}'
                    response = response.replace(md_table, html_table)
            
            # Clean up any remaining technical language and mentions
            # Remove any JSON mentions
            response = re.sub(r'Here\'s the JSON data for visualization:.*?(?=\n\n|\Z)', '', response, flags=re.DOTALL | re.IGNORECASE)
            response = re.sub(r'I\'ve included the JSON data for.*?(?=\n\n|\Z)', '', response, flags=re.DOTALL | re.IGNORECASE)
            response = re.sub(r'The JSON data is provided.*?(?=\n\n|\Z)', '', response, flags=re.DOTALL | re.IGNORECASE)
            
            # Remove technical phrases and language
            technical_phrases = [
                r'according to (?:the|our) database',
                r'the data shows',
                r'based on the analysis',
                r'query results indicate',
                r'the results show',
                r'as per the query',
                r'from the data',
                r'in the database',
                r'from our records',
                r'the query shows',
                r'database analysis',
                r'sql query',
                r'table',
                r'join',
                r'query',
                r'select',
                r'where',
                r'group by',
                r'order by',
                r'database schema'
            ]
            
            for phrase in technical_phrases:
                response = re.sub(phrase, '', response, flags=re.IGNORECASE)
            
            # Format the main response - any text comes first
            # Clean whitespace and formatting
            response = response.strip()
            
            # Remove any remaining HTML tags from the response
            response = re.sub(r'<.*?>', '', response)
            
            # Remove any lines that could be SQL or technical explanations
            lines = response.split('\n')
            clean_lines = []
            for line in lines:
                if not any(tech in line.lower() for tech in ['select', 'from', 'where', 'join', 'group by', 'database', 'query', 'sql']):
                    clean_lines.append(line)
            
            response = '\n'.join(clean_lines).strip()
            
            # Display the result
            with st.chat_message("assistant"):
                # Display the main conversational answer
                st.markdown(f'<div class="answer-highlight">{response}</div>', unsafe_allow_html=True)
                
                # Display SQL query in a collapsible section
                if sql_query:
                    st.markdown(f"""
                    <div class="sql-section">
                        <div class="sql-header">
                            <i>⚙️</i> SQL Query Details (Technical)
                        </div>
                        <div class="sql-code">
                    {sql_query}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                elif response:  # If we have a response but no SQL was extracted
                    # Try to extract SQL from the agent's execution chain output
                    try:
                        # Look for SQL in the response with a different pattern
                        alt_sql_match = re.search(r'(?:SELECT|INSERT|UPDATE|DELETE|CREATE|ALTER|DROP|WITH|SHOW)(?:.|\n)*?(?:;|LIMIT)', response, re.IGNORECASE)
                        if alt_sql_match:
                            alt_sql = alt_sql_match.group(0)
                            st.markdown(f"""
                            <div class="sql-section">
                                <div class="sql-header">
                                    <i>⚙️</i> SQL Query Details (Technical)
                                </div>
                                <div class="sql-code">
                            {alt_sql}
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            # If no SQL found, show a message about technical details
                            st.markdown(f"""
                            <div class="sql-section">
                                <div class="sql-header">
                                    <i>⚙️</i> Technical Details
                                </div>
                                <div class="sql-code">
                            The system executed a query to retrieve this information from our movie database.
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                    except Exception as e:
                        st.warning(f"Could not extract SQL query details: {str(e)}")
            
            # Add assistant message to history
            st.session_state.messages.append(AIMessage(content=response))
            
            # Create visualization based on data and type
            if wants_viz and json_data:
                visualization = create_visualization(json_data, viz_type)
                if visualization:
                    st.plotly_chart(visualization, use_container_width=True)
                else:
                    st.warning("Could not create visualization from the provided data.")
            elif wants_viz:
                st.warning("I couldn't generate a visualization for this query. Please try a different question or specify the type of chart you'd like to see.")
    
    # Sidebar with information
    with st.sidebar:
        st.header("About the Sakila Database")
        st.markdown("""
        <div class="sidebar-card">
        <p>The Sakila database represents a DVD rental store, featuring films, actors, customers, and rental information.</p>
        
        <h3>Example Questions:</h3>
        <ul>
        <li>Which movies are the most rented?</li>
        <li>How many customers have rented movies from New York?</li>
        <li>Who are the top 5 actors based on film appearances?</li>
        <li>What is the average rental duration for comedy films?</li>
        <li>Which customers spent the most in 2005?</li>
        <li>Show me a pie chart of films by rating</li>
        <li>What's the revenue breakdown by film category?</li>
        <li>Give me a bar chart of the most popular actors</li>
        <li>How many rentals were made per month in 2005 as a line chart?</li>
        <li>Which film length category has the most rentals?</li>
        <li>Show me the average rental rate by film rating as a visualization</li>
        <li>Display the customer distribution by country</li>
        <li>How many films are in each language?</li>
        <li>What's our inventory breakdown by store?</li>
        <li>Show the rental duration distribution across all films</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.header("Database Schema")
        st.markdown('<div class="schema-container">', unsafe_allow_html=True)
        st.image("https://dev.mysql.com/doc/sakila/en/images/sakila-schema.png", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True) 