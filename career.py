import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

# Constants
USER_DATA_FILE = "user_data.csv"
CAREER_DATA_FILE = "career.csv"

# Load the data from the CSV file
@st.cache_data
def load_data(file_name):
    try:
        # Load the CSV file
        df = pd.read_csv(file_name)
        return df
    except FileNotFoundError:
        st.error(f"Error: Could not find the data file '{file_name}'.")
        return None

# Function to create a new user
def create_user(username, password, name, email, age):
    # Check if the user data file exists
    if not os.path.exists(USER_DATA_FILE):
        # Create a new DataFrame
        df = pd.DataFrame(columns=["Username", "Password", "Name", "Email", "Age"])
    else:
        # Load existing user data
        df = pd.read_csv(USER_DATA_FILE)
    
    # Check if username already exists
    if (df['Username'] == username).any():
        st.error('Username already exists. Please choose another one.')
        return False

    # Create a new DataFrame with the new user data
    new_user_df = pd.DataFrame([[username, password, name, email, age]], columns=["Username", "Password", "Name", "Email", "Age"])

    # Concatenate the new DataFrame with the existing DataFrame
    df = pd.concat([df, new_user_df], ignore_index=True)
    
    # Save the updated DataFrame to CSV
    df.to_csv(USER_DATA_FILE, index=False)
    st.success('Registration successful! You can now login.')
    
    # Redirect to the login page after successful registration
    st.session_state.page = "login"
    
    return True

# Function to check if the username and password are valid
def is_valid_credentials(username, password):
    user_df = load_data(USER_DATA_FILE)
    if user_df is not None:
        return (user_df['Username'] == username) & (user_df['Password'] == password)
    return False

# Function to show the login page
def show_login_page():
    st.title('Login')
    username = st.text_input('Username')
    password = st.text_input('Password', type='password')

    if st.button('Login'):
        if is_valid_credentials(username, password).any():
            st.session_state.logged_in = True
            st.session_state.username = username
            st.session_state.page = "recommendation"
            return
        else:
            st.error('Invalid username or password')

# Function to show the registration page
def show_registration_page():
    st.title('Register')
    username = st.text_input('Username')
    password = st.text_input('Password', type='password')
    name = st.text_input('Name')
    email = st.text_input('Email')
    age = st.number_input('Age', min_value=0, max_value=150)

    if st.button('Register'):
        success = create_user(username, password, name, email, age)
        if success:
            st.session_state.page = "login"

# Function to show the recommendation page
def show_recommendation_page():
    df = load_data(CAREER_DATA_FILE)
    if df is None:
        return

    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform(df['Description'])
    
    st.title('Career Recommendation System')
    st.write("Welcome to the Career Recommendation System! Please describe your interests and skills below.")

    user_input = st.text_area('Your Interests and Skills:')

    if st.button('Get Recommendation'):
        if user_input:
            recommended_careers = recommend_careers(user_input, df, tfidf_vectorizer, tfidf_matrix)
            st.success(f"Based on your input, we recommend the following careers:")
            for career in recommended_careers:
                st.write(f"- {career}")
        else:
            st.warning('Please enter some text before getting a recommendation.')

# Function to recommend careers based on user input using cosine similarity
def recommend_careers(user_input, df, tfidf_vectorizer, tfidf_matrix):
    user_tfidf = tfidf_vectorizer.transform([user_input])
    cosine_similarities = cosine_similarity(user_tfidf, tfidf_matrix)
    similar_career_indices = cosine_similarities.argsort(axis=1)[:, ::-1].flatten()
    recommended_careers = df.iloc[similar_career_indices[:5]]['Career'].tolist()
    return recommended_careers

# Streamlit app
def main():
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
        st.session_state.page = "register"

    if st.session_state.page == "login":
        show_login_page()
    elif st.session_state.page == "register":
        show_registration_page()
    elif st.session_state.page == "recommendation":
        show_recommendation_page()

if __name__ == "__main__":
    main()
