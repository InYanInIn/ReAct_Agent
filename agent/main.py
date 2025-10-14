import json
import os
import threading
import traceback

from agent.CodeAgent import get_fixed_code


def load_human_eval_problems(path):
    problems = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                problems.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON on line: {line.strip()} - {e}")
    return problems


def run_test_with_timeout(code_str: str, test_str: str, timeout=300) -> bool:
    """
    Execute the generated code and run its test block with a timeout.
    Returns True if all tests pass, False otherwise.
    """
    result = {"passed": False}

    def target():
        try:
            # Create a shared execution environment
            env = {}

            # Execute the candidate code (defines the function)
            exec(code_str, env)

            # Execute the test code (calls check(...))
            exec(test_str, env)

            result["passed"] = True  # If no exception, test passed
        except Exception:
            print("Test failed:")
            traceback.print_exc(limit=1)

    thread = threading.Thread(target=target)
    thread.start()
    thread.join(timeout)

    if thread.is_alive():
        print("⏳ Skipping test due to timeout")
        thread.join()  # Ensure thread cleanup
        return False

    return result["passed"]


def get_fixed_code_with_timeout(buggy_code: str, timeout=15) -> str:
    """
    Wrapper to run get_fixed_code with a timeout.
    Returns the fixed code if completed within the timeout, otherwise None.
    """
    result = {"fixed_code": None}

    def target():
        try:
            result["fixed_code"] = get_fixed_code(source_code=buggy_code)
        except Exception as e:
            print(f"Error in get_fixed_code: {e}")

    thread = threading.Thread(target=target)
    thread.start()
    thread.join(timeout)

    if thread.is_alive():
        print("⏳ Skipping due to timeout in get_fixed_code")
        thread.join()  # Ensure thread cleanup
        return None

    return result["fixed_code"]


if __name__ == "__main__":
    human_eval_path = "..\\data\\humaneval.jsonl"
    problems = load_human_eval_problems(human_eval_path)
    problems = problems[33:40]

    print(f"Loaded {len(problems)} Python problems")

    # total = len(problems)
    # passed = 0

    total = len(problems)+13
    passed = 7

    for i, problem in enumerate(problems):
        print(f"\n=== Problem {problem['task_id']} ===")

        # The original buggy or incomplete code
        buggy_code = problem["prompt"]
        print("Buggy code:")
        print(buggy_code)
        print("---------------")
        # Use your model or agent to generate/fix code
        fixed_code = get_fixed_code_with_timeout(buggy_code, timeout=360)
        if fixed_code is None:
            print(f"❌ Skipped {problem['task_id']} due to timeout in get_fixed_code")
            continue
        print("Fixed code:")
        print(fixed_code)
        print("---------------")
        # Run unit tests
        passed_flag = run_test_with_timeout(fixed_code, problem["test"])

        if passed_flag:
            passed += 1
            print(f"✅ Passed {problem['task_id']}")
        else:
            print(f"❌ Failed {problem['task_id']}")

    pass_at_1 = passed / total
    print(f"\n=== Results ===")
    print(f"Total: {total}, Passed: {passed}, pass@1 = {pass_at_1:.3f}")

    # buggy_code_str = """
    # def tricky_function(lst):
    #     # Intended behavior:
    #     # 1. Keep only numbers > 2
    #     # 2. Skip numbers if cumulative sum exceeds 100
    #     out = []
    #     total = 0
    #     for x in lst:
    #         out.append(x*x)
    #         total += x*x
    #     return out
    #
    # """
    # Running iteration example:
    # print(get_fixed_code(buggy_code_str))
