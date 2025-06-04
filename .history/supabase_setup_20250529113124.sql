CREATE TABLE users (
    user_id UUID PRIMARY KEY,
    username TEXT NOT NULL,
    email TEXT UNIQUE NOT NULL,
    password TEXT NOT NULL,
    registration_time TIMESTAMP NOT NULL,
    last_login TIMESTAMP
);