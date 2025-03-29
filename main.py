import os
import sys
from check_sakila import check_sakila_db
import subprocess

def main():
    # Check if the Sakila database is available
    print("Checking Sakila database...")
    sakila_available = check_sakila_db()
    
    if not sakila_available:
        print("\nSakila database is required to run this application.")
        choice = input("Do you want to set up the Sakila database now? (y/n): ")
        
        if choice.lower() == 'y':
            print("Running setup_sakila.py...")
            import setup_sakila
            setup_result = setup_sakila.setup_sakila_db()
            
            if not setup_result:
                print("Failed to set up Sakila database. Exiting.")
                sys.exit(1)
        else:
            print("Sakila database is required to run this application. Exiting.")
            sys.exit(1)
    
    # Check if OpenAI API key is set
    if not os.getenv('OPENAI_API_KEY') or os.getenv('OPENAI_API_KEY') == 'your_openai_api_key':
        api_key = input("Please enter your OpenAI API key: ")
        if not api_key:
            print("OpenAI API key is required to run this application. Exiting.")
            sys.exit(1)
        
        # Update .env file with the API key
        with open('.env', 'r') as file:
            env_content = file.readlines()
        
        with open('.env', 'w') as file:
            for line in env_content:
                if line.startswith('OPENAI_API_KEY='):
                    file.write(f'OPENAI_API_KEY={api_key}\n')
                else:
                    file.write(line)
        
        print("OpenAI API key has been saved to .env file.")
    
    # Run the Streamlit app
    print("Starting the Streamlit application...")
    subprocess.run(["streamlit", "run", "app.py"])

if __name__ == "__main__":
    main() 