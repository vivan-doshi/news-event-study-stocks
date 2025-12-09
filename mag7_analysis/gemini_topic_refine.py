import google.generativeai as genai
import json
import os

# --- 1. Configuration ---
#
# ⚠️ IMPORTANT: Set your API key as an environment variable
# or directly here for testing.
#
# genai.configure(api_key="YOUR_API_KEY")
#
# Or, better practice, from an environment variable:
api_key = ""
if not api_key:
    print("Error: GOOGLE_API_KEY environment variable not set.")
    # In a real app, you'd exit here. For this example, we'll use a placeholder.
    # exit() 
else:
    genai.configure(api_key=api_key)


# --- 2. Create a Sample JSON File (for this example) ---
# In your real use case, you'll already have this file.
'''
sample_data = {
    "project": "Sales Analysis Q4",
    "report_type": "summary",
    "data": [
        {"region": "North", "sales": 150000, "reps": 10},
        {"region": "South", "sales": 120000, "reps": 8},
        {"region": "East", "sales": 180000, "reps": 12},
        {"region": "West", "sales": 135000, "reps": 9}
    ]
}
'''

data = None
with open('topic_terms_kmeansV2 2.json', 'r') as f:
    data = json.load(f)

print(data)
print(type(data))
'''
# --- 3. Load Your JSON Data ---
try:
    with open("data.json", "r") as f:
        json_content = json.load(f)
    
    # Convert the Python dictionary back into a string for the prompt
    json_string = json.dumps(json_content, indent=2)

except FileNotFoundError:
    print("Error: data.json not found.")
    json_string = None
except json.JSONDecodeError:
    print("Error: Failed to decode JSON from data.json.")
    json_string = None
'''

# --- 4. Define Your Prompt and Call the API ---
if data:
    # Initialize the model
    # You can also use 'gemini-1.5-flash' for faster responses
    model = genai.GenerativeModel('models/gemini-pro-latest') 

    # This is the crucial part:
    # Build a prompt that contains both your instructions
    # and the JSON data as a text string.
    
    prompt_template = f"""
    Here is a python dictionary containing topic id as key and an array of label terms 
    as value. Please analyze it and do the following:
    
    1.  Infer all the labels within the array values.
    2.  We have a total of 50 topic ids. I want to reduce this to 15 final labels 
    which are distinct domain representatives. For eg. Mergers & Acquisition, Innovation,
    CEO News, Existing Product Updates, Legal Changes, Layoffs & Recessions etc.
    3.  Give me a one to one mapping for the 50 topic IDs to these final 15 labels.

    Here is the python dictionary:
    
    ```python
    {data}
    ```
    
    Please provide the answers clearly.
    """

    print("--- Sending Prompt to Gemini ---")
    print(prompt_template)
    print("---------------------------------")
    
    try:
        # Send the prompt to the model
        response = model.generate_content(prompt_template)
        
        print("\n--- Gemini's Response ---")
        print(response)
        print("---------------------------")

    except Exception as e:
        print(f"An error occurred while calling the API: {e}")


