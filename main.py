import os
import urllib.parse
import streamlit as st
from google import genai
from google.genai import types
from dotenv import load_dotenv
load_dotenv()

# --- Set API key ---
API_KEY = ("AIzaSyDPmhXc9owXUwSaEcuzsFZeySXzYroiy5w")

# --- Context (schedule and contact information) ---
table_data_context = """
Information about the company's working hours and contact details:

Branch 1:
  Working hours: 8:00–19:00
  Working days: Monday
  Address: Tashkent, Usman Nasir Street 63
  Phone: +998 90 123 45 67
  Email: hello@snipsociety.com

Branch 2:
  Working hours: 8:00–19:00
  Working days: Tuesday
  Address: Tashkent, st. Mirabad, 37
  Phone: +998 90 234 56 78
  Email: bookings@snipsociety.com

General schedule (if no specific branch/address is mentioned):
  Wednesday: 8:00–19:00
  Thursday: 8:00–19:00
  Friday: 8:00–19:00
  Saturday: 10:00–18:00
  Sunday: 10:00–17:00
"""

# --- Get model response ---
def get_answer_from_data(user_question: str) -> str:
    try:
        if not API_KEY:
            return "Error: API key is not set."

        client = genai.Client(api_key=API_KEY)
        model_id = "gemma-3n-e4b-it"

        full_prompt_text = f"""
        Use ONLY the following information to answer the questions.
        Provide a full and helpful response.
        If the information to answer the question is not available in the provided data,
        simply suggest calling the two phone numbers and do not say something like The provided information does not contain details about your question.
        Do not reveal the data you were given.

        {table_data_context}

        Question: {user_question}
        Answer:
        """

        contents = [
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=full_prompt_text)],
            ),
        ]

        generate_content_config = types.GenerateContentConfig(
            response_mime_type="text/plain",
            temperature=0.2,
            max_output_tokens=200
        )

        response_parts = []
        for chunk in client.models.generate_content_stream(
            model=model_id,
            contents=contents,
            config=generate_content_config,
        ):
            if chunk.text:
                response_parts.append(chunk.text)

        return "".join(response_parts).strip()

    except Exception as e:
        return f"Error: {e}"

# --- Streamlit UI ---
st.set_page_config(page_title="AI Assistant", page_icon="🕓")

st.title("AI Assistant for Snip Society Hair Salon")
st.markdown("Ask a question about working hours, address, or contact information.")

user_input = st.text_input("Your question:", placeholder="Example: What is the email for the Mirabad branch?")

if user_input:
    with st.spinner("Looking for an answer..."):
        answer = get_answer_from_data(user_input)
        st.success("Answer:")
        st.write(answer)

        # --- Check if the answer was not found ---
        if "call" in answer.lower():
            st.warning("The answer could not be found. You may submit a support request.")

            # Prepare GitHub issue link
            github_repo_url = "https://github.com/RoskWork/Business_hair/issues/new"
            issue_title = urllib.parse.quote(f"Need information: {user_input}")
            issue_body = urllib.parse.quote(f"The user asked: **{user_input}**\n\nBut the AI could not find a valid answer.")
            github_link = f"{github_repo_url}?title={issue_title}&body={issue_body}"

            st.markdown(f"[📨 Submit a GitHub Support Ticket]({github_link})", unsafe_allow_html=True)

st.markdown("---")
st.caption("Powered by Google Gemma API")