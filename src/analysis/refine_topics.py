import google.generativeai as genai
import json
import os
import argparse
import logging
import sys

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def refine_labels(input_file, output_file, api_key=None):
    if not api_key:
        api_key = os.environ.get("GOOGLE_API_KEY")
    
    if not api_key:
        logger.error("GOOGLE_API_KEY not found in environment variables.")
        return

    genai.configure(api_key=api_key)
    
    logger.info(f"Loading raw clusters from {input_file}...")
    try:
        with open(input_file, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        logger.error(f"Input file not found: {input_file}")
        return

    # Prepare prompt
    prompt_template = f"""
    You are a financial news analyst expert. I have 5 clusters of financial news topics from "Mag7" stocks (Apple, Amazon, Google, etc.).
    Below is a JSON where keys are Cluster IDs and values contain a list of 'terms' (top keywords).
    
    TASK:
    Generate a distinct, professional 2-3 word label for each cluster ID based on its terms.
    The label must be specific (e.g., 'Regulatory Antitrust', 'AI Product Launch', 'Earnings Guidance', 'Macro Trade Policy').
    Avoid generic terms like 'Business' or 'General'.
    
    INPUT JSON:
    ```json
    {json.dumps(data, indent=2)}
    ```
    
    OUTPUT FORMAT:
    Return ONLY a valid JSON object mapping Cluster ID to the new Label. No markdown formatting, no explanations.
    Example:
    {{
        "0": "Label A",
        "1": "Label B"
    }}
    """

    logger.info("Sending prompt to Gemini...")
    model = genai.GenerativeModel('gemini-pro')
    
    try:
        response = model.generate_content(prompt_template)
        response_text = response.text.strip()
        
        # Clean up potential markdown code blocks if Gemini ignores instructions
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]
        
        cleaned_map = json.loads(response_text)
        logger.info(f"Generated Labels: {cleaned_map}")
        
        # Save
        with open(output_file, 'w') as f:
            json.dump(cleaned_map, f, indent=4)
        logger.info(f"Saved label map to {output_file}")
        
    except Exception as e:
        logger.error(f"Gemini API Error: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", default="reports/clustering/topics_5_clusters.json", help="Path to raw clusters JSON")
    parser.add_argument("--output_path", default="reports/clustering/topic_labels_map.json", help="Path to output labels JSON")
    args = parser.parse_args()
    
    refine_labels(args.input_path, args.output_path)
