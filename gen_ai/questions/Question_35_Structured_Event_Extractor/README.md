# Structured Event Extractor

A Python application that uses Google Gemini API to extract structured event information from college event announcements.

## Overview

This program demonstrates how to:
- Use the Google Gemini API for structured information extraction
- Work with JSON response schemas in Gemini
- Handle API errors gracefully
- Manage API keys securely using environment variables
- Process missing/incomplete data intelligently

## Features

✅ Extracts 5 key fields from event announcements:
- Event Name
- Date
- Time
- Venue
- Registration Link

✅ Handles missing information gracefully (converts to `null`)

✅ Comprehensive error handling for:
- Rate limits and quota errors
- Timeout errors
- Authentication failures
- Invalid API responses

✅ Beginner-friendly code with detailed comments

✅ Includes 3 test cases (including an edge case with missing time)

## Requirements

- Python 3.7+
- `google-genai` package
- `python-dotenv` package
- Valid Gemini API key

## Installation

### 1. Install Dependencies

```bash
pip install google-genai python-dotenv
```

### 2. Set Up API Key

1. Get your Gemini API key from: https://ai.google.dev/
2. Create a `.env` file in the project directory:
   ```
   GEMINI_API_KEY=your_api_key_here
   ```

**Important:** Never commit the `.env` file to version control!

## Usage

Run the program:

```bash
python event_extractor.py
```

### Expected Output

The program will process three test announcements and display:

```
==================================================
STRUCTURED EVENT EXTRACTOR
==================================================

==================================================
TEST CASE 1
==================================================

Announcement:
The Annual Technology Fest 2026 will be held on September 15, 2026 from 10:00 AM to 4:00 PM at the Main Auditorium. Students can register at https://college.edu/techfest

Extracted Event Details:
{
    "event_name": "Annual Technology Fest 2026",
    "date": "September 15, 2026",
    "time": "10:00 AM to 4:00 PM",
    "venue": "Main Auditorium",
    "registration_link": "https://college.edu/techfest"
}

[... TEST CASE 2 and 3 ...]

==================================================
PROGRAM FINISHED
==================================================
```

## Test Cases

### Test Case 1: Complete Event Information
All five fields are present in the announcement.

### Test Case 2: Edge Case - Missing Time
**This tests the requirement:** The announcement intentionally lacks a time field.
The output should contain `"time": null`

### Test Case 3: Complete Event Information
All five fields are present in the announcement.

## Code Structure

The program is organized into logical sections:

1. **API Setup** - Load and verify API key
2. **Schema Definition** - Define JSON response format
3. **Helper Functions** - Extract and display functions
4. **Test Data** - Three test announcements
5. **Main Program** - Process and display results

## Key Implementation Details

### Response Schema
- Uses `STRING` types (not union types like `["string", "null"]`)
- Compatible with `google-genai` 2.20.0+

### Missing Value Handling
Converts Gemini responses to JSON `null` for:
- "unspecified"
- "not specified"
- "not mentioned"
- "unknown"
- "n/a"
- "na"
- Empty strings

### Temperature Setting
Uses `temperature=0` for consistent, factual extraction (no creativity)

### JSON Output
Uses `json.dumps(..., indent=4)` for readable formatting

## Error Handling

The program handles:

| Error Type | Response |
|-----------|----------|
| Missing API key | Exits with error message |
| Invalid JSON | Displays parsing error |
| Rate limit (429) | Asks user to wait and retry |
| Timeout | Suggests checking internet connection |
| Authentication error (401/403) | Asks user to verify API key |
| Model not found (404) | Informs about endpoint issue |
| General API errors | Displays error details |

## For Viva Preparation

### Key Points to Explain:

1. **Why Gemini?** - Structured output, JSON schema support, cost-effective
2. **Why temperature=0?** - Consistent extraction, no hallucinations
3. **Security** - API key in .env, never hard-coded
4. **Schema** - Ensures consistent output format
5. **Edge Cases** - Test case 2 demonstrates missing data handling
6. **Error Handling** - Production-ready error messages

### Sample Questions & Answers:

**Q: Why use response schema?**
A: To ensure Gemini always returns the exact fields we need in JSON format.

**Q: How do we handle missing time in test case 2?**
A: Gemini returns null, and we don't invent the missing data. The final JSON correctly shows `"time": null`

**Q: Why load API key from .env?**
A: Security best practice - never hard-code credentials in source code.

**Q: What does temperature=0 do?**
A: Ensures deterministic, factual extraction without creative responses.

## Limitations

- Only processes text announcements (no image parsing)
- Limited to the 5 predefined fields
- Requires valid Gemini API key
- Depends on Gemini's ability to understand announcement context

## Future Enhancements

- Add support for multiple languages
- Extract additional fields (speaker info, capacity, etc.)
- Batch processing multiple announcements
- Database storage of extracted events
- Web interface for easy access
- Calendar integration

## Troubleshooting

### "GEMINI_API_KEY not found"
- Ensure `.env` file exists in the project directory
- Verify the environment variable name is exactly `GEMINI_API_KEY`

### "Invalid JSON response"
- Check your internet connection
- Verify your API key is valid
- Try again (may be a temporary API issue)

### "Rate limit exceeded"
- Wait a few minutes before retrying
- Consider upgrading your Gemini API plan

## License

This is an academic assignment implementation.

## Author

College Assignment - Question 35: Structured Event Extractor
