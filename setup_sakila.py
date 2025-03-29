import os
import sys
import urllib.request
import zipfile
import mysql.connector
from mysql.connector import Error
import tempfile
from dotenv import load_dotenv

def setup_sakila_db():
    """Download and set up the Sakila database"""
    load_dotenv()
    
    print("Setting up Sakila database...")
    
    # MySQL credentials
    mysql_host = os.getenv('MYSQL_HOST', 'localhost')
    mysql_user = os.getenv('MYSQL_USER', 'root')
    mysql_password = os.getenv('MYSQL_PASSWORD', 'Admin@123')
    
    # Create a temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        # Download Sakila database files
        print("Downloading Sakila database files...")
        sakila_zip_url = "https://downloads.mysql.com/docs/sakila-db.zip"
        sakila_zip_path = os.path.join(temp_dir, "sakila-db.zip")
        
        try:
            urllib.request.urlretrieve(sakila_zip_url, sakila_zip_path)
        except Exception as e:
            print(f"Error downloading Sakila database: {e}")
            return False
        
        # Extract files
        print("Extracting files...")
        try:
            with zipfile.ZipFile(sakila_zip_path, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
        except Exception as e:
            print(f"Error extracting files: {e}")
            return False
        
        # Paths to SQL files
        schema_file = os.path.join(temp_dir, "sakila-db", "sakila-schema.sql")
        data_file = os.path.join(temp_dir, "sakila-db", "sakila-data.sql")
        
        # Check if files exist
        if not os.path.exists(schema_file) or not os.path.exists(data_file):
            print("Error: SQL files not found in the downloaded package.")
            return False
        
        # Read SQL files
        try:
            with open(schema_file, 'r', encoding='utf-8') as f:
                schema_sql = f.read()
            
            with open(data_file, 'r', encoding='utf-8') as f:
                data_sql = f.read()
        except Exception as e:
            print(f"Error reading SQL files: {e}")
            return False
        
        # Connect to MySQL and import the Sakila database
        try:
            connection = mysql.connector.connect(
                host=mysql_host,
                user=mysql_user,
                password=mysql_password
            )
            
            if connection.is_connected():
                cursor = connection.cursor()
                
                # Check if Sakila database already exists
                cursor.execute("SHOW DATABASES LIKE 'sakila'")
                result = cursor.fetchone()
                
                if result:
                    choice = input("Sakila database already exists. Do you want to overwrite it? (y/n): ")
                    if choice.lower() != 'y':
                        print("Setup cancelled.")
                        return False
                    
                    # Drop existing database
                    print("Dropping existing Sakila database...")
                    cursor.execute("DROP DATABASE sakila")
                
                # Execute schema SQL
                print("Creating Sakila database schema...")
                for statement in schema_sql.split(';'):
                    if statement.strip():
                        cursor.execute(statement.strip() + ';')
                
                # Execute data SQL
                print("Importing Sakila data...")
                for statement in data_sql.split(';'):
                    if statement.strip():
                        try:
                            cursor.execute(statement.strip() + ';')
                        except Error as e:
                            print(f"Warning: Error executing statement: {e}")
                            continue
                
                connection.commit()
                print("✅ Sakila database setup complete!")
                return True
        
        except Error as e:
            print(f"Error connecting to MySQL: {e}")
            return False
        finally:
            if 'connection' in locals() and connection.is_connected():
                cursor.close()
                connection.close()

if __name__ == "__main__":
    setup_sakila_db() 