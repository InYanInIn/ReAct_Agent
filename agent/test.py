from agent.workflow import get_fixed_code

def read_multiline_input(prompt="Enter your code snippet (end with empty line):"):
    print(prompt)
    lines = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        if line.strip() == "":  # empty line means "done"
            break
        lines.append(line)
    return "\n".join(lines)

flag = 0

while flag != "2":

    print("Menu:")
    print("1. Test another code snippet")
    print("2. Exit")
    flag = input("Choose an option(1,2): ")
    if flag == "1":
        code = read_multiline_input()
        print("Wait, it may take a few minutes...")
        fixed = get_fixed_code(code)
        print("-----------")
        print(fixed)
    elif flag == "2":
        print("Exiting...")
        break
    else:
        print("Invalid option. Please choose 1 or 2.")

