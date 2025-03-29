import os
import mysql.connector
from mysql.connector import Error
from dotenv import load_dotenv

def check_sakila_db():
    """Check if the Sakila database exists and is accessible"""
    load_dotenv()
    
    try:
        # Connect to MySQL
        connection = mysql.connector.connect(
            host=os.getenv('MYSQL_HOST', 'localhost'),
            user=os.getenv('MYSQL_USER', 'root'),
            password=os.getenv('MYSQL_PASSWORD', 'Admin@123')
        )
        
        if connection.is_connected():
            cursor = connection.cursor()
            
            # Check if Sakila database exists
            cursor.execute("SHOW DATABASES LIKE 'sakila'")
            result = cursor.fetchone()
            
            if result:
                print("✅ Sakila database found!")
                
                # Check if some key tables exist
                connection.database = 'sakila'
                cursor.execute("SHOW TABLES LIKE 'film'")
                film_table = cursor.fetchone()
                
                if film_table:
                    print("✅ Film table found. Sakila database appears to be properly installed.")
                    return True
                else:
                    print("❌ Film table not found. Sakila database may not be properly installed.")
                    show_installation_instructions()
                    return False
            else:
                print("❌ Sakila database not found.")
                show_installation_instructions()
                return False
                
    except Error as e:
        print(f"❌ Error connecting to MySQL: {e}")
        show_installation_instructions()
        return False
    finally:
        if 'connection' in locals() and connection.is_connected():
            cursor.close()
            connection.close()

def show_installation_instructions():
    """Display instructions for installing the Sakila database"""
    print("\n=== Sakila Database Installation Instructions ===")
    print("The Sakila database is required for this application to work.")
    print("To install the Sakila database:")
    
    print("\n=== Option 1: Using MySQL Workbench ===")
    print("1. Open MySQL Workbench")
    print("2. Connect to your local MySQL server")
    print("3. Go to 'Server' > 'Data Import'")
    print("4. Select 'Import from Self-Contained File'")
    print("5. Download the Sakila database files from: https://dev.mysql.com/doc/index-other.html")
    print("6. Follow the import wizard to complete the installation")
    
    print("\n=== Option 2: Using Command Line ===")
    print("1. Download the Sakila database files from: https://dev.mysql.com/doc/index-other.html")
    print("2. Extract the downloaded files")
    print("3. Open a terminal/command prompt")
    print("4. Navigate to the directory containing the extracted files")
    print("5. Run the following commands:")
    print("   a. mysql -u root -p < sakila-schema.sql")
    print("   b. mysql -u root -p < sakila-data.sql")
    print("   (Enter your MySQL password when prompted)")
    
    print("\n=== Option 3: Running setup_sakila.py ===")
    print("This application includes a setup script that can automatically download and install the Sakila database.")
    print("Run: python setup_sakila.py")
    
    print("\nAfter installation, verify that the Sakila database is installed correctly by running this script again.")

if __name__ == "__main__":
    check_sakila_db() 