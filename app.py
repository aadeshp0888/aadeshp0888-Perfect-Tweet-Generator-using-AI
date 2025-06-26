
import streamlit as st
import os
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

load_dotenv()
try:
    huggingface_api_key = os.environ['HUGGINGFACE_API_KEY']
    client = InferenceClient(token=huggingface_api_key)
except KeyError:
    client = None

# CORE AI LOGIC 

def generate_tweets(topic, client_instance):
    prompt = f"""
    You are an expert social media manager specializing in writing viral tweets.
    Your task is to generate 3 to 5 distinct and engaging tweets based on a given topic.

    Here are the rules you must follow:
    1.  **Quantity:** Generate between 3 and 5 tweets.
    2.  **Character Limit:** Each tweet must be under 280 characters.
    3.  **Hashtags:** Include 2-3 relevant and popular hashtags in each tweet.
    4.  **Tone:** The tone should be engaging, catchy, and modern. Use emojis where appropriate.
    5.  **Formatting:**
        -   Start each tweet with "Tweet X:" (e.g., "Tweet 1:", "Tweet 2:").
        -   Separate each tweet with a "---" line for clarity.

    **Topic to write about:** "{topic}"
    """
    try:
        response = client_instance.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            model="meta-llama/Meta-Llama-3-8B-Instruct",
            max_tokens=512,
            temperature=0.7,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"An error occurred with the API call: {e}"

# 4. STREAMLIT USER INTERFACE (All changes are here!)


## UI ENHANCEMENT: Set a wider page layout for a more modern feel
st.set_page_config(layout="wide", page_title="Viral Tweet Generator", page_icon="🐦")

## UI ENHANCEMENT: Use st.sidebar for a cleaner look
with st.sidebar:
    st.title("🐦 Viral Tweet Generator")
    st.markdown("Craft compelling tweets in seconds. Enter a topic and let the AI do the magic!")
    
    # API Key Check
    if client is None:
        st.error("Hugging Face API key not found! Set HUGGINGFACE_API_KEY in your .env file.", icon="🚨")
        st.stop()
    else:
        st.success("API key loaded successfully!", icon="✅")

    st.markdown("---")
    st.markdown("### ⚙️ How to Use")
    st.markdown("""
    1.  Enter a topic or a rough idea in the text box.
    2.  Click the 'Generate Tweets' button.
    3.  Copy your favorite tweets!
    """)
    st.markdown("---")
    st.markdown("Built by a curious AI explorer.") # Removed the specific backend name

# --- Main Page Layout ---
st.header("✍️ Your Tweet Crafting Studio")

# ## UI ENHANCEMENT: Use columns to separate input from output area
col1, col2 = st.columns([0.8, 1.2]) # Create two columns with different widths

with col1:
    st.subheader("Your Idea")
    user_topic = st.text_area(
        "Enter your topic or idea here:",
        height=150,
        placeholder="e.g., Why learning a new skill is crucial in 2024"
    )

    if st.button("✨ Generate Tweets", type="primary", use_container_width=True):
        if user_topic:
            with st.spinner("AI is thinking... this might take a moment... 🤔"):
                # Store the generated tweets in session state
                st.session_state.generated_tweets = generate_tweets(user_topic, client)
        else:
            st.error("Please enter a topic first!", icon="⚠️")

with col2:
    st.subheader("AI-Generated Tweets")
    # ## UI ENHANCEMENT: Use session state to keep the output on screen
    if 'generated_tweets' in st.session_state and st.session_state.generated_tweets:
        # ## UI ENHANCEMENT: Create a container with a border for the output
        with st.container(border=True):
            st.markdown(st.session_state.generated_tweets)
    else:
        # ## UI ENHANCEMENT: Display a friendly placeholder
        st.info("Your generated tweets will appear here once you click the button.", icon="👇")
