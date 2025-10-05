import re
import os
import datetime
import glob
import json

def extract_score_from_response(response_text):
    if not isinstance(response_text, str):
        print("Error: Input must be a string.")
        return None
    match = re.search(r'\{\s*"?(?:Score)"?\s*:\s*(\d+)\s*\}', response_text)
    if match:
        try:
            score_str = match.group(1)
            score_int = int(score_str)
            if 1 <= score_int <= 5:
               return score_int
            else:
               print(f"Error: Extracted score {score_int} is outside the expected range (1-5).")
               return None
        except ValueError:
            print("Error: Found pattern but the score value is not a valid integer.")
            return None
        except Exception as e:
            print(f"Error during extraction: {e}")
            return None
    else:
        try:
            data = json.loads(response_text)
            if isinstance(data, dict) and "Score" in data and isinstance(data["Score"], int):
                if 1 <= data["Score"] <= 5:
                    return data["Score"]
                else:
                    print(f"Error: Extracted score {data['Score']} is outside the expected range (1-5).")
                    return None
            else:
                print("Error: Could not find the specified JSON pattern {\"Score\": <integer>} using regex, and fallback JSON parsing failed or format is incorrect.")
                return None
        except json.JSONDecodeError:
            print("Error: Could not find the specified JSON pattern {\"Score\": <integer>} using regex, and the response is not valid JSON for fallback.")
            return None
        except Exception as e:
            print(f"Error during fallback parsing: {e}")
            return None
        
def extract_json_from_text(text: str, key = None):
    matches = re.findall(r'\{.*?\}', text, re.DOTALL)
    if matches:
        if key:
            for match in matches:
                if key in match:
                    return match
        return matches[-1]
    return None

def is_template_string(s: str) -> bool:
    try:
        s.format()
        return False
    except KeyError:
        return True
    
def extract_time(file_path):
    file_name = os.path.basename(file_path)
    time_str = file_name.split(".json")[0]
    return datetime.strptime(time_str, "%Y-%m-%d-%H-%M")

def find_latest_json(result_dir):
    pattern = os.path.join(result_dir, "*-*-*-*-*.json")
    files = glob.glob(pattern)
    if not files:
        print(f"Json file not found at {result_dir}")
        return None
    latest_file = max(files, key=extract_time)
    return latest_file

def extract_choice_answer(text: str):
    match = re.search(r"['\"]?Answer['\"]?:\s*['\"]?(A|B|C|D)['\"]?", text)
    if match:
        return match.group(1)
    return None

def choice_correctness(answer, gt: str):
    if answer in ["A", "B", "C", "D"]:
        answer = {"Answer": answer}
    else:
        try:
            answer = extract_choice_answer(answer)
        except Exception as e:
            print(f"Unexpected error in choice_correctness: {e}")
            return 0
    if answer:
        if str(answer).upper() == gt.upper():
            correctness = 1
        else:
            correctness = 0
    else:
        correctness = 0
    return correctness

def fix_json_format(json_str: str):
    try:
        # 将所有的单引号替换为双引号
        fixed_json_str = re.sub(r"(?<=:)\s*'([^']*)'", r': "\1"', json_str)  # 键和值的单引号修复
        fixed_json_str = re.sub(r"'([^']*)'", r'"\1"', fixed_json_str)  # 字符串中的单引号修复
        return fixed_json_str
    except Exception as e:
        print(f"Error in fix_json_format: {e}")
        return None
    
def main():
    string = "To clearly distinguish between the two energy levels, we need to ensure that the energy difference is at least as large as the uncertainty in energy, which is related to the lifetime of the states. The uncertainty principle in energy is given by \u0394E*t >= h/4\u03c0, where \u0394E is the uncertainty in energy, t is the lifetime, and h is the Planck constant.\n\nFor the first state with a lifetime of 10^-9 sec, we can calculate the minimum uncertainty in energy as:\n\n\u0394E1 = h / (4\u03c0*t1) = (6.626e-34 J*s) / (4\u03c0*10^-9 s) = 5.28e-26 J\n\nConverting this to eV, we get:\n\n\u0394E1 \u2248 3.3e-8 eV\n\nFor the second state with a lifetime of 10^-8 sec, we can calculate the minimum uncertainty in energy as:\n\n\u0394E2 = h / (4\u03c0*t2) = (6.626e-34 J*s) / (4\u03c0*10^-8 s) = 5.28e-27 J\n\nConverting this to eV, we get:\n\n\u0394E2 \u2248 3.3e-10 eV\n\nTo clearly distinguish between the two energy levels, the energy difference should be at least as large as the larger of the two uncertainties. Therefore, the energy difference should be at least \u0394E2 \u2248 3.3e-10 eV.\n\nComparing this with the given options, the correct answer is:\n\n{\"Answer\": B}"
    choice_correctness(string, "C")

if __name__ == "__main__":
    main()