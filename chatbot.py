import streamlit as st
from transformers import pipeline

# Load the Hugging Face model for question-answering
qa_pipeline = pipeline('question-answering', model='distilbert-base-cased-distilled-squad')

# Set up the Streamlit app
st.title("Educational Chatbot")
st.write("Ask me any question related to education!")

# User input for the question
question = st.text_input("Your question:", placeholder="What is education?")

# Provide a default context, or you can extend it to dynamic context loading
default_context = """
Education is the process of facilitating learning or the acquisition of knowledge, skills, values, beliefs, and habits.
Educational methods include teaching, training, storytelling, discussion, and directed research. Education frequently takes place
under the guidance of educators, but learners may also educate themselves. Education can take place in formal or informal settings,
and any experience that has a formative effect on the way one thinks, feels, or acts may be considered educational.
"""

# Optional user-provided context (advanced feature)
use_custom_context = st.checkbox("Use custom context?")
context = default_context
if use_custom_context:
    context = st.text_area("Provide context for the question:", default_context)

# Process the question when the user clicks the button
if st.button("Get Answer"):
    if question:
        # Use the model to generate the answer
        result = qa_pipeline(question=question, context=context)
        answer = result['answer']

        # Display the result
        st.write(f"**Answer:** {answer}")
    else:
        st.write("Please ask a question to get an answer.")

# Footer
st.write("Powered by Hugging Face Transformers")
