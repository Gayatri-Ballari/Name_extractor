import json
import os
import re
import ast
from dotenv import load_dotenv
from langchain_groq import ChatGroq

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

def extract_list_from_text(text):
    match = re.search(r'\[.*?\]', text, re.DOTALL)  # Find the list enclosed in square brackets
    if match:
        try:
            return ast.literal_eval(match.group())  # Convert string representation of list to an actual list
        except (SyntaxError, ValueError):
            return []  # Return an empty list if there's a parsing error
    return []  # Return an empty list if no list is found

def extract_company_names(text):
    """
    Extracts company names from a given text using Groq's Mistral model (with multilingual support).

    Parameters:
        text (str): The input text for Named Entity Recognition (NER).

    Returns:
        str: Extracted company name(s) or an empty string if none found.
    """

    # Ensure input text is not empty or just whitespace
    if not text.strip():
        return ""  # Return empty string if no meaningful input is given

    # Initialize Groq model
    model = 'mixtral-8x7b-32768'
    llm = ChatGroq(model_name=model)

    # Multilingual NER prompt
    prompt = (
        "<s>[INST] Extract company names from the following text using Named Entity Recognition (NER) across multiple languages. "
        "Return the company name(s) in a list format. If a company name appears in an abbreviated form (e.g., AMZ for Amazon), return the full company name only. "
        "If no company name is found, return an empty list '[]'. No explanation is needed. "
        "Recognize company names in any language, including English, Spanish, French, German, Chinese, Japanese, and others. "
        "Ensure that company names are correctly formatted according to their official representation. "
        "If uncertain, make an educated guess based on common business names.\n\n"
        f'Text: "{text}" [/INST]'
    )

    try:
        # Invoke model
        response = llm.invoke(prompt)
        company_name = response.content.strip()

        return company_name if company_name else ""

    except Exception as e:
        return f"Error calling Groq API: {e}"


# Example Usage
if __name__ == "__main__":
    text = "What is the latest revenue report for DR. LAVA TUDO?"
    result = extract_company_names(text)
    result = extract_list_from_text(result)
    print(result)  # Should print the company name list
