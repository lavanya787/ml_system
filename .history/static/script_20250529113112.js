function showRegister() {
    document.getElementById('login-container').style.display = 'none';
    document.getElementById('register-container').style.display = 'flex';
}

function showLogin(showMessage = false) {
    document.getElementById('register-container').style.display = 'none';
    document.getElementById('login-container').style.display = 'flex';
    const successMessage = document.getElementById('success-message');
    if (setMessage) {
        successMessage.textContent = 'Successfully Registered';
        successMessage.style.display = 'block';
    } else {
        successMessage.style.display = 'none';
    }
}

async function handleRegister() {
    const username = document.getElementById('register-username').value;
    const email = document.getElementById('register-email').value;
    const password = document.getElementById('register-password').value;

    const response = await fetch('/api/register', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username, email, password })
    });
    const result = await response.json();
    
    if (response.ok) {
        showLogin(true); // Show "Successfully Registered" message
    } else {
        alert(result.message); // Show error message if registration fails
    }
}

async function handleLogin() {
    const email = document.getElementById('login-email').value;
    const password = document.getElementById('login-password').value;

    const response = await fetch('/api/login', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email, password })
    });
    const result = await response.json();
    
    if (response.ok) {
        // Hide the "SIGN UP" section after successful login
        document.getElementById('signup-section').style.display = 'none';
        localStorage.setItem('user_id', result.user_id);
        window.location.href = '/chatbot';
    } else {
        alert(result.message); // Show error message if login fails
    }
}