import re
import json
from typing import Union

def extract_json_from_text(text: str, type = "list") -> Union[dict, list, None]:
    if not isinstance(text, str):
        print("Input must be a string.")
        return None
    
    cleaned_text = text.strip().replace('\ufeff', '')

    try:
        if type == "list":
            pattern = r'```(?:json)?\s*([\s\S]*?)\s*```'
        elif type == "dict":
            pattern = r'```(?:json)?\s*({.*?})\s*```'
        match = re.search(pattern, cleaned_text, re.IGNORECASE)
        if not match:
            print("No JSON code block found (supports ``` and ```json).")
            return None
        json_candidate = match.group(1).strip()
    except Exception as e:
        print(f"Code block extraction failed: {e}")
        return None
            
    try:
        data = json.loads(json_candidate)
        if not isinstance(data, dict):
            print("Extracted content is not a dictionary.")
            return None
        return data
    except Exception as e1:
        try:
            json_fixed = (
                json_candidate
                .replace("'", '"')
                .replace("\n", "")
            )
            data = json.loads(json_fixed)
            if not isinstance(data, dict):
                print("Content after simple fix is still not a dictionary.")
                return None
            return data
        except Exception as e2:
            print(f"JSON parsing failed!\nOriginal error: {e1}\nError after fix: {e2}\nContent:\n{json_candidate}")
            return None
        
        
def extract_choice_from_response(response, choices: list) -> str:
    """Use same choice extraction logic as the environment"""
    import re
    
    # First try JSON format
    try:
        json_pattern = r'\{[^}]*"choice"\s*:\s*"([^"]+)"[^}]*\}'
        json_match = re.search(json_pattern, response, re.IGNORECASE)
        if json_match:
            choice = json_match.group(1).lower()
            if choice in [c.lower() for c in choices]:
                return choice
    except:
        pass
    
    # Try dictionary format
    try:
        dict_pattern = r'"?choice"?\s*:\s*"?([^",\s]+)"?'
        dict_match = re.search(dict_pattern, response, re.IGNORECASE)
        if dict_match:
            choice = dict_match.group(1).lower()
            if choice in [c.lower() for c in choices]:
                return choice
    except:
        pass

    return "Error: Fail to extract choice"