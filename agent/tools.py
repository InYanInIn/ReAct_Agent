import ast
import json
import subprocess, tempfile, os, re


def run_in_sandbox(llm_output: str, timeout=30) -> dict:
    """
    Test validation
    """
    py_code = extract_python_code(llm_output)

    # создаём временный файл с кодом
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(py_code)
        path = f.name

    success = False
    stdout = ""

    try:
        result = subprocess.run(
            ["python", path],
            capture_output=True,
            text=True,
            timeout=timeout
        )
        success = result.returncode == 0
        stdout = result.stdout if success else result.stderr

    except Exception as e:
        stdout = str(e)
    finally:
        os.remove(path)

    return {"success": success, "stdout": stdout}

def extract_python_code(llm_output: str) -> str:
    cleaned = re.sub(r"<think>.*?</think>", "", llm_output, flags=re.DOTALL | re.IGNORECASE)

    pattern = r"```python\n(.*?)```"
    match = re.search(pattern, cleaned, re.DOTALL)
    if match:
        return match.group(1).strip()
    else:
        # Если нет блока, возвращаем всё (но лучше предупреждать)
        return llm_output.strip()


def output_post_processing(text: str, max_len: int = 4000) -> str:
    """
    Remove ReAct/assistant thinking blocks and other noisy markers that should not be passed
    to downstream tools. Returns a cleaned, trimmed string.
    Comments in English as requested.
    """
    if not isinstance(text, str):
        return str(text)

    # 1) Remove explicit <think>...</think> or <think> ... </think> style blocks (DOTALL)
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)

    # 2) Remove common labeled sections like "Thought:", "Observation:", "Action:", "Final Answer:" etc.
    cleaned = re.sub(r"(?m)^(Thought|Observation|Action|Action Input|Final Answer|Reasoning):.*$", "", cleaned)

    # 4) Remove multiple contiguous newlines and trim whitespace
    cleaned = re.sub(r"\n{2,}", "\n", cleaned).strip()

    # 5) Optionally truncate to avoid extremely long prompts to downstream tools
    if len(cleaned) > max_len:
        cleaned = cleaned[:max_len].rsplit("\n", 1)[0]  # cut at last newline boundary

    return cleaned.strip()

def save_history_to_logs(path: str, history):
    # print(type(history))
    if isinstance(history, list):
        try:
            # Load existing history if the file exists
            if os.path.exists(path):
                try:
                    with open(path, 'r') as log_file:
                        existing_history = json.load(log_file)
                    if not isinstance(existing_history, list):
                        existing_history = []
                except json.JSONDecodeError:
                    # Handle empty or invalid JSON file
                    existing_history = []
            else:
                existing_history = []

            # Append the new history
            existing_history.extend(history)

            # Save the updated history back to the file
            with open(path, 'w') as log_file:
                json.dump(existing_history, log_file, indent=4)
        except Exception as e:
            print(f"An error occurred while saving history: {e}")
