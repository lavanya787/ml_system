function showRegister() {
    document.getElementById('login-container').style.display = 'none';
    document.getElementById('register-container').style.display = 'flex';
}

function showLogin() {
    document.getElementById('register-container').style.display = 'none';
    document.getElementById('login-container').style.display = 'flex';
}

// Function to show notification
function showNotification(message, callback) {
    const notification = document.getElementById('notification');
    const notificationMessage = document.getElementById('notification-message');
    notificationMessage.textContent = message;
    notification.style.display = 'flex';
    // Hide after 2 seconds and call the callback
    setTimeout(() => {
        notification.style.display = 'none';
        if (callback) callback(); // Execute callback after notification hides
    }, 2000);
}

async function handleRegister() {
    const username = document.getElementById('register-username').value;
    const email = document.getElementById('register-email').value;
    const password = document.getElementById('register-password').value;

    try {
        const response = await fetch('/api/register', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ username, email, password })
        });
        const result = await response.json();
        
        if (response.ok) {
            console.log('Registration successful, redirect URL:', result.redirect); // Debug log
            showNotification('Welcome! Your registration was successful. Let\'s get started.', () => {
                window.location.href = result.redirect;  // Redirect to Streamlit app
            });
        } else {
            console.error('Registration failed:', result.message); // Debug log
            showNotification(result.message); // Show error message (e.g., "User already exists")
        }
    } catch (error) {
        console.error('Error during registration:', error); // Debug log
        showNotification('Error during registration');
    }
}

async function handleLogin() {
    const email = document.getElementById('login-email').value;
    const password = document.getElementById('login-password').value;

    try {
        const response = await fetch('/api/login', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ email, password })
        });
        const result = await response.json();
        
        if (response.ok) {
            console.log('Login successful, redirect URL:', result.redirect); // Debug log
            showNotification('Welcome! You have successfully logged in.', () => {
                window.location.href = result.redirect;  // Redirect to Streamlit app
            });
        } else {
            console.error('Login failed:', result.message); // Debug log
            showNotification(result.message); // Show error message (e.g., "Incorrect password")
        }
    } catch (error) {
        console.error('Error during login:', error); // Debug log
        showNotification('Error during login');
    }
}