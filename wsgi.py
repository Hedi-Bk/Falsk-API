import os 
from app import app 
from dotenv import load_dotenv

load_dotenv()

if __name__ == '__main__':
    port =int(os.environ.get('PORT', 4000))
    app.run(host='0.0.0.0', port=port)