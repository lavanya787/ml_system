from flask import Flask, request, jsonify, render_template, redirect, url_for
from flask_cors import CORS
from supabase import create_client, Client
import bcrypt
import uuid
from datetime import datetime

app = Flask(__name__)
CORS(app)

# Supabase setup
SUPABASE_URL = "https://uctyxchurvievzvhthru.supabase.co"
SUPABASE_KEY = "your-supabase-key"
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Serve the main page
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/register', methods=['POST'])
def register():
    data = request.get_json()
    username = data.get('username')
    email = data.get('email')
    password = data.get('password')

    # Check if user already exists
    user = supabase.table('users').select('*').eq('email', email).execute()
    if user.data:
        return jsonify({'message': 'User already exists'}), 400

    # Hash password
    hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

    # Generate unique user ID
    user_id = str(uuid.uuid4())

    # Store user in Supabase
    registration_time = datetime.utcnow().isoformat()
    new_user = {
        'user_id': user_id,
        'username': username,
        'email': email,
        'password': hashed_password,
        'registration_time': registration_time
    }
    supabase.table('users').insert(new_user).execute()

    return jsonify({'message': 'Registration successful'}), 200

@app.route('/api/login', methods=['POST'])
def login():
    data = request.get_json()
    email = data.get('email')
    password = data.get('password')

    # Fetch user from Supabase
    user = supabase.table('users').select('*').eq('email', email).execute()
    if not user.data:
        return jsonify({'message': 'User not found'}), 404

    user_data = user.data[0]
    # Verify password
    if not bcrypt.checkpw(password.encode('utf-8'), user_data['password'].encode('utf-8')):
        return jsonify({'message': 'Incorrect password'}), 401

    # Update last login time
    login_time = datetime.utcnow().isoformat()
    supabase.table('users').update({'last_login': login_time}).eq('email', email).execute()

    return jsonify({'message': 'Login successful', 'user_id': user_data['user_id']}), 200

# Placeholder route for chatbot (after successful login)
@app.route('/chatbot')
def chatbot():
    return "Welcome to the Chatbot! (Placeholder)"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)