# Company name extraction using Groq model

## Overview
This project evaluates Company Name extraction i.e Named Entity Recognition(NER) model predictions against ground truth labels with the help of the prompt we have written. It utilizes OpenAI for Groq model invocation and entity extraction. The evaluation includes precision, recall, F1-score matrix, Fuzzy score calculations, along with visualization of metrics.

## Features
- Company name extractions(NER model)
- OpenLLM for invoking Groq models
- Synthetic test case execution from a CSV file
- Calculating a evalution metrics (precision, recall, F1-score, Fuzzy score)
- Visualization of evaluation metrics 
- Unit test using Pytest
- --for extracting a company name
- --no company name in the user query
- --handling the abbrevations

## Prerequisites
- Python
- Groq API
- Dependencies: Install using the command below:
  ```sh
  pip install -r requiremnets.txt
  ```
- `testdata.csv` containing test cases
- `.env` containing Groq API key.

## Project Structure
```
.
├── validation.py               # Main script to run evaluation
├── validation_metrics.py       # Contains entity extraction logic
├── testdata.csv                # CSV file with synthetic test cases
├── README.md                   # Project documentation
└── requirements.txt            # Python dependencies
```

## How It Works
1. **Entity Extraction**
   - Uses openLLM to invoke a foundation model for extracting company names.
   - The `extract_company_names()` function calls using Groq model.

2. **Entity Evaluation**
   - Answer the user question by the `prompt` about company name extractions.
   - Compares extracted company names against ground truth labels.

4. **Synthetic Test Execution**
   - Loads test cases from `testdata.csv`.
   - Runs extraction and evaluation on each test case.

5. **Visualization**
   - Computes mean and median scores.
   - Generates bar charts for performance analysis.
   - It also computes the fuzzy score for each test cases.
   - generate the distribution graph for analysis.

6. **Unit Testing**
   - Uses Pytest to validate the test cases of the entity extraction function.

## Langchain Integration
The script uses Groq's Mistral model to extract company names from text. 

### Steps for Integration:
1. Ensure Groq API key is configured in `.env`.
2. Modify `validation.py` to include Groq API calls.
3. Example Groq invocation:
   ```python
   from langchain_groq import ChatGroq

    # Load environment variables
   load_dotenv()
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    os.environ["GROQ_API_KEY"] = GROQ_API_KEY
       prompt = (
           "<s>[INST] Extract company names from the following text..."
           f'Text: "{text}" [/INST]'
       )
         model = 'mixtral-8x7b-32768'
        llm = ChatGroq(model_name=model)
    try:
        # Invoke model
        response = llm.invoke(prompt)
        company_name = response.content.strip()

        return company_name if company_name else ""

    except Exception as e:
        return f"Error calling Groq API: {e}"
   ```

4. The `extract_list_from_text()` function extracts a valid list format from the response.

## Results
- The script prints extracted entities and evaluation metrics and the test caes for the testing of script.
- A bar chart is generated comparing mean and median scores.
- A distribution graph will be generated to analyze the fuzzy score of each use cases.

## Usage
### Running the Evaluation
```sh
python vaidation_metrics.py
```
This will:
- Run synthetic tests using the test dataset
- Evaluate the extracted entities
- Generate visual metrics

### Running Unit Tests
```sh
pytest case_test.py
```
This will execute predefined unit tests to validate the entity extraction logic.

## Example Usage
```python
text = "I work for Msft and also for amz"
result = extract_company_names(text)
result = extract_list_from_text(result)
print(result)  # Expected: ["Microsoft", "Amazon"]
```

