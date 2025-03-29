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
import base64

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
        background-color: #f5f7f9;
    }
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .css-1d391kg, .css-12oz5g7 {
        padding: 2rem 1rem;
    }
    .stButton>button {
        background-color: #4e7afc;
        color: white;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        font-weight: 500;
    }
    .stTextInput>div>div>input {
        border-radius: 5px;
    }
    .chat-message {
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    .chat-message.user {
        background-color: #e6f3ff;
    }
    .chat-message.assistant {
        background-color: white;
    }
    h1 {
        color: #1e3a8a;
    }
    .sql-code {
        background-color: #f8f9fa;
        border-radius: 5px;
        padding: 1rem;
        font-family: 'Courier New', monospace;
        margin: 1rem 0;
        border-left: 3px solid #4e7afc;
        white-space: pre-wrap;
        overflow-x: auto;
    }
    .stAlert {
        border-radius: 5px;
    }
    .annotation-container {
        margin-top: 1rem;
        padding: 1rem;
        background-color: #f0f8ff;
        border-radius: 5px;
        border-left: 3px solid #4e7afc;
    }
    .result-table {
        width: 100%;
        border-collapse: collapse;
        margin: 1rem 0;
        font-size: 0.9em;
        font-family: sans-serif;
        box-shadow: 0 0 20px rgba(0, 0, 0, 0.15);
        border-radius: 5px;
        overflow: hidden;
    }
    .result-table thead tr {
        background-color: #4e7afc;
        color: #ffffff;
        text-align: left;
    }
    .result-table th,
    .result-table td {
        padding: 12px 15px;
    }
    .result-table tbody tr {
        border-bottom: 1px solid #dddddd;
    }
    .result-table tbody tr:nth-of-type(even) {
        background-color: #f3f3f3;
    }
    .result-table tbody tr:last-of-type {
        border-bottom: 2px solid #4e7afc;
    }
    .explanation {
        background-color: #f0f8ff;
        padding: 1rem;
        border-radius: 5px;
        margin-top: 1rem;
        border-left: 3px solid #4e7afc;
    }
    pre {
        white-space: pre-wrap;
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
            try:
                result, wants_table, wants_viz, viz_type = process_query(agent, user_input)
                
                # Extract SQL query
                sql_match = re.search(r'```sql\n(.*?)\n```', result, re.DOTALL)
                sql_query = ""
                if sql_match:
                    sql_query = sql_match.group(1)
                    # Remove SQL block from the main result
                    result = re.sub(r'```sql\n.*?\n```', '', result, flags=re.DOTALL)
                
                # Extract JSON data for visualization if requested
                viz_data = None
                if wants_viz:
                    viz_data, json_str = extract_json_data(result)
                    if json_str:
                        # Remove JSON block from the main result
                        result = re.sub(r'```json\s*.*?\s*```', '', result, flags=re.DOTALL)
                
                # Process the result to make it more conversational
                # Handle table formatting differently if user wants tables
                table_html = ""
                if wants_table:
                    # Prepare the final display text
                    # Get text before the first table (if any)
                    intro_text = result
                    
                    # First, extract the table from the result
                    # Look for proper markdown tables
                    table_pattern = r'(\|.+\|[\r\n]+\|[-| :]+\|[\r\n]+((?:\|.+\|[\r\n]+)+))'
                    table_match = re.search(table_pattern, result, re.DOTALL)
                    
                    if table_match:
                        table_markdown = table_match.group(0)
                        # Remove the table from the intro text
                        intro_text = result.replace(table_markdown, '')
                        
                        # Split the table into rows
                        rows = table_markdown.strip().split('\n')
                        
                        if len(rows) >= 3:  # Need header, separator, and at least one data row
                            header_row = rows[0]
                            data_rows = rows[2:]  # Skip the separator row
                            
                            # Parse header cells
                            headers = []
                            for cell in header_row.split('|')[1:-1]:  # Skip the first and last empty elements
                                headers.append(cell.strip())
                            
                            # Create HTML table - using compact format without newlines to prevent escaping
                            table_html = '<div class="result-table-container"><table class="result-table"><thead><tr>'
                            
                            for header in headers:
                                table_html += f'<th>{header}</th>'
                                
                            table_html += '</tr></thead><tbody>'
                            
                            # Process data rows
                            for row in data_rows:
                                cells = [cell.strip() for cell in row.split('|')[1:-1]]
                                if len(cells) == len(headers):
                                    table_html += '<tr>'
                                    for cell in cells:
                                        table_html += f'<td>{cell}</td>'
                                    table_html += '</tr>'
                            
                            table_html += '</tbody></table></div>'
                    
                    # Clean up the intro text
                    intro_text = intro_text.strip()
                
                else:
                    # If not wanting tables, remove them completely
                    intro_text = re.sub(r'\|.+\|[\r\n]+\|[-| :]+\|[\r\n]+((?:\|.+\|[\r\n]+)+)', '', result, flags=re.DOTALL)
                
                # Remove phrases that indicate technical explanations
                technical_phrases = [
                    r'Here\'s how I arrived at this answer:.*?(?=\n\n|\Z)',
                    r'Based on the SQL query.*?(?=\n\n|\Z)',
                    r'The SQL query.*?(?=\n\n|\Z)',
                    r'This query.*?(?=\n\n|\Z)',
                    r'Let me explain how.*?(?=\n\n|\Z)',
                    r'To answer this question, I.*?(?=\n\n|\Z)',
                    r'I ran a SQL query.*?(?=\n\n|\Z)',
                    r'Here\'s what the data shows:.*?(?=\n\n|\Z)',
                    r'According to the database.*?(?=\n\n|\Z)',
                    r'From the results.*?(?=\n\n|\Z)',
                    r'Looking at the data.*?(?=\n\n|\Z)',
                    r'I found.*?(?=\n\n|\Z)',
                    r'When I queried.*?(?=\n\n|\Z)'
                ]
                
                for phrase in technical_phrases:
                    intro_text = re.sub(phrase, '', intro_text, flags=re.DOTALL | re.IGNORECASE)
                
                # Clean up any remaining technical language
                intro_text = re.sub(r'SQL|query|database|table schema|join|select|from|where|group by|order by|having', '', intro_text, flags=re.IGNORECASE)
                
                # Only remove bullet points if not in table mode
                if not wants_table:
                    # Remove any bullet points or numbering
                    intro_text = re.sub(r'^\s*[\*\-]\s+', '', intro_text, flags=re.MULTILINE)
                    intro_text = re.sub(r'^\s*\d+\.\s+', '', intro_text, flags=re.MULTILINE)
                
                # Clean up multiple newlines and whitespace
                intro_text = re.sub(r'\n{2,}', ' ', intro_text)
                intro_text = re.sub(r'\s{2,}', ' ', intro_text)
                intro_text = intro_text.strip()
                
                # Display the result
                with st.chat_message("assistant"):
                    # Create a clean, conversational display with a friendly avatar - without multi-line strings
                    intro_container = f'<div class="chat-bubble" style="display:flex; align-items:flex-start; margin-bottom:1rem; background-color:#f0f7ff; padding:1rem; border-radius:12px;"><div style="width:40px; height:40px; border-radius:50%; background-color:#4e7afc; color:white; display:flex; align-items:center; justify-content:center; margin-right:12px; flex-shrink:0; font-size:20px;">🎬</div><div style="font-size:1.1em; line-height:1.4;">{intro_text}</div></div>'
                    
                    # Display the conversational part first
                    st.markdown(intro_container, unsafe_allow_html=True)
                    
                    # If we have a table, display it separately
                    if table_html:
                        st.markdown(f'<div style="margin-top: 1rem;">{table_html}</div>', unsafe_allow_html=True)
                    
                    # If we have visualization data, display the chart
                    if wants_viz and viz_data:
                        try:
                            st.subheader("Visualization")
                            fig = create_visualization(viz_data, viz_type)
                            if fig:
                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.warning("Could not create visualization from the provided data.")
                        except Exception as viz_error:
                            st.error(f"Error creating visualization: {str(viz_error)}")
                    elif wants_viz:
                        st.warning("I couldn't generate a visualization for this query. Please try a different question or specify the type of chart you'd like to see.")
                    
                    # Display SQL details separately if we have them
                    if sql_query:
                        st.markdown(f'<details><summary style="cursor: pointer; color:#888; font-size:0.8em; margin-top:0.5rem;">Admin: SQL Details</summary><div style="background-color:#f8f9fa; padding:0.5rem; border-radius:5px; font-family:monospace; white-space:pre-wrap; font-size:0.8em;">{sql_query}</div></details>', unsafe_allow_html=True)
                
                # Add assistant message to history (only the conversational part plus a note about tables/visualizations)
                history_content = intro_text
                if table_html:
                    table_note = "\n\n[Results shown in table format]"
                    history_content += table_note
                
                if wants_viz and viz_data:
                    viz_note = "\n\n[Results visualized as a chart]"
                    history_content += viz_note
                
                st.session_state.messages.append(AIMessage(content=history_content))
                
            except Exception as e:
                error_message = f"Sorry, I couldn't find an answer to that question. Could you try asking in a different way?"
                with st.chat_message("assistant"):
                    st.error(error_message)
                    st.markdown(f"<details><summary>Error details (for troubleshooting)</summary><pre>{str(e)}</pre></details>", unsafe_allow_html=True)
                st.session_state.messages.append(AIMessage(content=error_message))
    
    # Sidebar with information
    with st.sidebar:
        st.header("About the Sakila Database")
        st.markdown("""
        <div style="background-color: white; padding: 1.5rem; border-radius: 10px; box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);">
        The Sakila database is a sample database representing a DVD rental store, featuring films, actors, 
        customers, and rental information.
        
        <h3>Example Questions:</h3>
        <ul>
        <li>Which movies are the most rented?</li>
        <li>How many customers have rented movies from New York?</li>
        <li>Who are the top 5 actors based on film appearances?</li>
        <li>What is the average rental duration for comedy films?</li>
        <li>Which customers spent the most in 2005?</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.header("Database Schema")
        st.image("https://dev.mysql.com/doc/sakila/en/images/sakila-schema.png", use_column_width=True)

# Check if the user wants a visualization
def detect_visualization_request(query):
    viz_keywords = {
        "bar chart": "bar",
        "bar graph": "bar",
        "barchart": "bar",
        "bargraph": "bar",
        "pie chart": "pie",
        "piechart": "pie",
        "line chart": "line",
        "line graph": "line",
        "linechart": "line",
        "linegraph": "line",
        "visualization": "auto",
        "visualisation": "auto",
        "visualize": "auto",
        "visualise": "auto",
        "chart": "auto",
        "graph": "auto",
        "plot": "auto"
    }
    
    query_lower = query.lower()
    for keyword, viz_type in viz_keywords.items():
        if keyword in query_lower:
            return True, viz_type
    
    return False, None

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
    You are a helpful and friendly database assistant for a movie rental store. Your name is MovieBot. 

    Answer the following question about the Sakila movie rental database:
    "{modified_input}"

    Important guidelines:
    1. You are speaking directly to a customer with no technical knowledge
    2. Be friendly, helpful, and personable - use a conversational tone
    3. Be concise - keep answers to 1-3 sentences whenever possible
    4. Include real data from the database in your answer (numbers, names, etc.)
    5. NEVER mention SQL, queries, databases, tables, or any technical terms
    6. NEVER explain your reasoning or methodology
    7. Respond as if you're a friendly store employee helping a customer
    """
    
    if wants_table:
        prompt_template += """
        8. VERY IMPORTANT: Since the user requested tabular data, you MUST format your response using markdown table syntax. 
        Format your data like this:
        
        | Column1 | Column2 | Column3 |
        |---------|---------|---------|
        | Value1 | Value2 | Value3 |
        | Value4 | Value5 | Value6 |
        
        Make sure the table is properly formatted with the separator line (|---|---|) and consistent column counts.
        """
    
    if wants_viz:
        prompt_template += f"""
        8. VERY IMPORTANT: Since the user requested a visualization, you MUST include a properly formatted JSON array in your response that I can use to create a {viz_type if viz_type != 'auto' else ''} chart.
        
        Format your data like this (enclosed in triple backticks with 'json' language specifier):
        
        ```json
        [
          {{"label": "Category1", "value": 42}},
          {{"label": "Category2", "value": 28}},
          {{"label": "Category3", "value": 15}}
        ]
        ```
        
        The JSON MUST be an array of objects with consistent keys across all objects. Label the keys appropriately based on the data (e.g., "movie_title", "rental_count", "category", "amount", etc.).
        
        If the visualization is time-based, include a "date" or "time" field in your JSON.
        If it's a comparison, use "label" for the category name and "value" for the numeric value.
        
        DO NOT OMIT THIS JSON DATA - it is required for visualization.
        """
    
    prompt_template += """
    Remember to be friendly, direct, and non-technical. Just answer the question like a helpful person would.
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
                    y=df.columns[1],
                    title=f"Trend of {df.columns[1].replace('_', ' ').title()} by {df.columns[0].replace('_', ' ').title()}"
                )
        else:
            # Default to bar chart
            fig = px.bar(df, x=df.columns[0], y=df.columns[1])
        
        # Improve layout
        fig.update_layout(
            margin=dict(l=20, r=20, t=40, b=20),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0.02)",
            font_family="Arial",
            title_font_size=16,
            xaxis_title_font_size=14,
            yaxis_title_font_size=14
        )
        
        return fig
        
    except Exception as e:
        st.error(f"Error creating visualization: {str(e)}")
        return None 