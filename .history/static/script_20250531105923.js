function showLogin() {
    document.getElementById('register-form').style.display = 'none';
    document.getElementById('login-form').style.display = 'block';
}

function showRegister() {
    document.getElementById('login-form').style.display = 'none';
    document.getElementById('register-form').style.display = 'block';
}

async function register() {
    const username = document.getElementById('reg-username').value;
    const email = document.getElementById('reg-email').value;
    const password = document.getElementById('reg-password').value;
    const messageDiv = document.getElementById('message');

    try {
        const response = await fetch('/api/register', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ username, email, password }),
        });
        const result = await response.json();
        if (response.ok) {
            messageDiv.style.color = 'green';
            messageDiv.textContent = result.message;
            // Redirect to Streamlit app
            window.location.href = result.redirect;
        } else {
            messageDiv.style.color = 'red';
            messageDiv.textContent = result.message;
        }
    } catch (error) {
        messageDiv.style.color = 'red';
        messageDiv.textContent = 'Error during registration';
    }
}

async function login() {
    const email = document.getElementById('login-email').value;
    const password = document.getElementById('login-password').value;
    const messageDiv = document.getElementById('message');

    try {
        const response = await fetch('/api/login', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ email, password }),
        });
        const result = await response.json();
        if (response.ok) {
            messageDiv.style.color = 'green';
            messageDiv.textContent = result.message;
            // Redirect to Streamlit app
            window.location.href = result.redirect;
        } else {
            messageDiv.style.color = 'red';
            messageDiv.textContent = result.message;
        }
    } catch (error) {
        messageDiv.style.color = 'red';
        messageDiv.textContent = 'Error during login';
    }
}