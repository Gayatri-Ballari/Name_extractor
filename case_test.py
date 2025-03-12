import pytest
from validation import extract_company_names  # Replace with your actual script filename
# from datetime import datetime, UTC

# # Ensure the current datetime is timezone-aware
# datetime_now = datetime.now(UTC)

def test_extract_valid_company():
    """Test extracting a valid company name from input."""
    qn1 = "I work for Amazon"
    ans1 = '* Amazon'
    result1 = extract_company_names(qn1, credentials_file="aws_credentials.json")
    assert result1.strip() == ans1, f"Expected {ans1}, but got {result1}"

def test_extract_no_company():
    """Test when no company name is mentioned."""
    qn2 = "I work "
    ans2 = '[]' # Assuming the function should return an empty string when no company is found
    result2 = extract_company_names(qn2, credentials_file="aws_credentials.json")
    assert result2.strip() == ans2, f"Expected '{ans2}', but got '{result2}'"

def test_extract_abbreviated_company():
    """Test extracting a company name from an abbreviation."""
    qn3 = "I work for Msft and also for amz"
    ans3 = "['Microsoft', 'Amazon']" # Assuming the function can recognize 'Amz' as 'Amazon'
    result3 = extract_company_names(qn3, credentials_file="aws_credentials.json")
    assert result3.strip() == ans3, f"Expected {ans3} but got {result3}"

if __name__ == "__main__":
    pytest.main()
