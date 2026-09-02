from google import genai
from dotenv import load_dotenv
import os
import json

# ============================================================
# 1. LOAD API KEY SECURELY
# ============================================================
# Load environment variables from .env file
load_dotenv()

# Get API key from environment variable (never hard-code it!)
api_key = os.getenv("GEMINI_API_KEY")

# Verify API key exists
if not api_key:
    print("Error: GEMINI_API_KEY not found in .env file.")
    print("Please create a .env file with: GEMINI_API_KEY=your_key_here")
    exit()

# ============================================================
# 2. CREATE GEMINI CLIENT
# ============================================================
# Initialize the Gemini API client with the secure API key
client = genai.Client(api_key=api_key)


# ============================================================
# 3. DEFINE RESPONSE SCHEMA
# ============================================================
# This schema tells Gemini to return JSON with these fields
# Using STRING types (not union types like ["string", "null"])
event_schema = {
    "type": "OBJECT",
    "properties": {
        "event_name": {"type": "STRING"},
        "date": {"type": "STRING"},
        "time": {"type": "STRING"},
        "venue": {"type": "STRING"},
        "registration_link": {"type": "STRING"}
    },
    "required": ["event_name", "date", "time", "venue", "registration_link"]
}

# ============================================================
# 4. DEFINE MISSING VALUE INDICATORS
# ============================================================
# Common ways Gemini indicates missing information
missing_values = [
    "unspecified",
    "not specified",
    "not mentioned",
    "unknown",
    "n/a",
    "na",
    "null",
    ""
]

# ============================================================
# 5. DEFINE REQUIRED FIELDS
# ============================================================
required_fields = ["event_name", "date", "time", "venue", "registration_link"]

# ============================================================
# 6. HELPER FUNCTION: EXTRACT EVENT INFORMATION
# ============================================================
def extract_event_info(announcement):
    """
    Extract structured event information from an announcement.
    
    Args:
        announcement (str): The event announcement text
        
    Returns:
        dict: Extracted event data with null values for missing fields
    """
    
    # Check if input is empty
    if not announcement.strip():
        print("Error: Announcement cannot be empty.")
        return None
    
    # Create the extraction prompt for Gemini
    prompt = f"""
You are a structured event information extractor.

Extract EXACTLY these five fields from the college event announcement:

1. event_name
2. date
3. time
4. venue
5. registration_link

RULES:
- Return only a valid JSON object.
- Use EXACTLY these five field names.
- If a field is missing or not mentioned, return null (not the string "null", but JSON null).
- Never invent or guess information.
- If the time is not mentioned, return null for time.
- Do NOT add explanations or markdown.
- Do NOT add any text outside the JSON object.

College event announcement:
{announcement}
"""
    
    try:
        # Call Gemini API with structured output
        response = client.models.generate_content(
            model="gemini-3.6-flash",
            contents=prompt,
            config={
                "temperature": 0,  # Zero temperature for consistent extraction
                "response_mime_type": "application/json",
                "response_schema": event_schema
            }
        )
        
        # Get response text
        result_text = response.text.strip()
        
        # Check for empty response
        if not result_text:
            print("Error: Gemini returned an empty response.")
            return None
        
        # Parse JSON response
        event_data = json.loads(result_text)
        
        # Ensure all required fields exist
        for field in required_fields:
            if field not in event_data:
                event_data[field] = None
        
        # Convert missing-value strings to JSON null
        for field in required_fields:
            value = event_data.get(field)
            
            if value is None:
                event_data[field] = None
            elif isinstance(value, str):
                if value.strip().lower() in missing_values:
                    event_data[field] = None
        
        return event_data
    
    # Handle JSON parsing errors
    except json.JSONDecodeError:
        print("Error: Gemini did not return valid JSON.")
        return None
    
    # Handle API errors
    except Exception as e:
        error_message = str(e).lower()
        
        # Rate limit / Quota exceeded
        if "429" in error_message or "resource_exhausted" in error_message:
            print("Error: Gemini API rate limit or quota exceeded.")
            print("Please wait and try again later.")
        
        # Timeout error
        elif "timeout" in error_message or "timed out" in error_message:
            print("Error: Gemini API request timed out.")
            print("Please check your internet connection and try again.")
        
        # Model not found
        elif "404" in error_message or "not_found" in error_message:
            print("Error: Gemini model or API endpoint was not found.")
        
        # Authentication error
        elif "401" in error_message or "403" in error_message:
            print("Error: API authentication failed.")
            print("Please verify your GEMINI_API_KEY in the .env file.")
        
        # Other API errors
        else:
            print("Error: Gemini API request failed.")
            print("Details:", str(e))
        
        return None


# ============================================================
# 7. HELPER FUNCTION: DISPLAY TEST RESULTS
# ============================================================
def display_results(test_num, announcement, event_data):
    """Display test case results in formatted output."""
    print("\n" + "=" * 50)
    print(f"TEST CASE {test_num}")
    print("=" * 50)
    print("\nAnnouncement:")
    print(announcement.strip())
    print("\nExtracted Event Details:")
    if event_data:
        print(json.dumps(event_data, indent=4))
    else:
        print("Failed to extract information.")


# ============================================================
# 8. THREE TEST ANNOUNCEMENTS
# ============================================================
test_announcements = [
    # TEST CASE 1: All fields available
    """The Annual Technology Fest 2026 will be held on September 15, 2026 from 10:00 AM to 4:00 PM at the Main Auditorium. Students can register at https://college.edu/techfest""",
    
    # TEST CASE 2: TIME IS MISSING (REQUIRED EDGE CASE)
    """The AI and Machine Learning Workshop will be conducted on October 10, 2026 at the Computer Science Seminar Hall. Students can register at https://college.edu/ai-workshop""",
    
    # TEST CASE 3: All fields available
    """A Career Guidance Seminar will be organized on November 5, 2026 from 2:00 PM to 5:00 PM at the University Conference Hall. Registration is available at https://college.edu/career-seminar"""
]



# ============================================================
# 9. MAIN PROGRAM
# ============================================================
def main():
    """Main function to run the Structured Event Extractor."""
    
    # Display program header
    print("\n" + "=" * 50)
    print("STRUCTURED EVENT EXTRACTOR")
    print("=" * 50)
    
    # Process each test announcement
    for test_num, announcement in enumerate(test_announcements, start=1):
        event_data = extract_event_info(announcement)
        display_results(test_num, announcement, event_data)
    
    # Display program completion message
    print("\n" + "=" * 50)
    print("PROGRAM FINISHED")
    print("=" * 50 + "\n")


# ============================================================
# 10. PROGRAM ENTRY POINT
# ============================================================
if __name__ == "__main__":
    main()