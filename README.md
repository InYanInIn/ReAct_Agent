# Code Fixing Agent

This project is a Python-based code-fixing agent that analyzes, diagnoses, and fixes Python code snippets. It uses a combination of tools and a language model to identify issues in the code, propose fixes, and validate them.

## Features
- Analyze Python code for bugs and issues.
- Automatically generate fixes for the identified problems.
- Validate the fixes using a sandboxed environment.
- Interactive mode for testing code snippets.
- Batch processing for datasets of code problems.
- Utilizes the **ReAct-style agent** for reasoning and acting in iterative steps.
- Powered by **Qwen3-0.6B**, a lightweight language model, for code analysis and generation.
- Built on **LangGraph**, a graph-based framework for managing state transitions and tool interactions.
---

## Testing Results

The agent was evaluated on a Python subset of HumanEvalFix with 40 records. Due to the time required to process the entire dataset, the results were divided into two batches:

### Batch 1: Records 1–20
- **Total:** 20  
- **Passed:** 9  
- **Failures:** 10  
- **Timeouts:** 1  
- **Pass@1:** 0.450  

### Batch 2: Records 21–40
- **Total:** 20  
- **Passed:** 11  
- **Failures:** 8  
- **Timeouts:** 1  
- **Pass@1:** 0.550  

### Overall Results
- **Total Successes:** 20  
- **Total Failures:** 18  
- **Total Timeouts:** 2  
- **Overall Pass@1:** 0.500

### Note on Model Performance

For a lightweight and less powerful model like **Qwen3-0.6B**, these results are quite impressive. However, the model demonstrates significant struggles with more complex tasks, particularly those involving:

- Nested structures, such as brackets `()` or `{}`.
- Tasks requiring precise character counting or manipulation.
- Problems with intricate logical dependencies.

While the model performs well on simpler tasks, its limitations become evident as task complexity increases.

## Installation
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-folder>
    ```
2. Install the required dependencies:
   ```bash
    pip install -r requirements.txt
    ```
<hr></hr>

## Usage
### Command-Line Interface
1. Run the agent/test.py script:  
    ```bash
    python agent/test.py
    ```
2. Follow the prompts:
   - Enter your buggy Python code snippet.
   - Wait for the agent to analyze and propose fixes.
   - Review the proposed fixes and see if they resolve the issues.

<hr></hr>

### Running On a Dataset
1. Place your dataset in JSONL format (one JSON object per line) at data/humaneval.jsonl. 
Each JSON object should have the following structure:  
    ```
    json{
        "task_id": "unique_id",
        "prompt": "buggy_code_here",
        "test": "test_code_here"
    }
    ```
2. Run the agent/main.py script with the dataset path:
    ```bash
    python agent/main.py
    ```
   
3. The script will:
    - Load the dataset.
    - Attempt to fix each code snippet.
    - Validate the fixes using the provided test cases.
    - Output the results, including the pass rate.

<hr></hr>

## Example
### Dataset mode
1. Input:
    ```
    json{
            "task_id": "example_1",
            "prompt": "def add(a, b): return a - b",
            "test": "assert add(2, 3) == 5"
        }
    ```
2. Output:
    ```plaintext
    Loaded 1 Python problem
    === Problem Python/0 ===
    Buggy code:
    def add(a, b): return a - b
    ---------------
    Fixed code:
    def add(a, b): return a + b
    ---------------
    ✅ Passed Python/0
    ===Results===
    Total: 1, Passed: 1, pass@1: 1.000
    ```