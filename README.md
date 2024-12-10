/opt/homebrew/bin/python3 -m venv myvenv
source myvenv/bin/activate
pip install --no-cache-dir -r requirements.txt 
python func/tps.py
deactivate


