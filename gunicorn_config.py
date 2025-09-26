# gunicorn_config.py

# Number of worker processes
workers = 4

# The socket to bind to.
# Bind to 127.0.0.1:5001 to ensure the app is only accessible from the server itself.
bind = '0.0.0.1:5001'

# The user to run the Gunicorn processes as.
# Replace 'n8nadmin' if you use a different user.
# user = 'n8nadmin'

# 1. Use the Gthread worker class for multi-threading
worker_class = 'gthread' 

# 2. Set the number of threads per worker. 
# Total concurrency = workers (4) * threads (10) = 40 parallel requests.
# This maximizes I/O handling.
threads = 10 

# 3. Increase timeout (The default 30s is too short for premium scrapes + model inference)
timeout = 120 
