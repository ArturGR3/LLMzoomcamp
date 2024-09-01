import threading

def execute_with_timeout(code: str, timeout: int = 10):
    result = [None, None]  # [execution_result, error_message]
    
    def run_code():
        try:
            exec(code)
            # Store the result (assumes the last variable is the answer)
            result[0] = locals().get('answer', None)
        except Exception as e:
            result[1] = str(e)

    thread = threading.Thread(target=run_code)
    thread.start()
    thread.join(timeout)

    if thread.is_alive():
        return None, "Execution timed out"
    
    return result[0], result[1]

# Example 1: Simple calculation
code1 = """
def calculate_area(length, width):
    return length * width

answer = calculate_area(10, 5)
"""

# Example 2: Infinite loop
code2 = """
while True:
    pass
"""

# Example 3: Code with an error
code3 = """
def divide(a, b):
    return a / b

answer = divide(10, 0)
"""

# Run examples
examples = [
    ("Simple calculation", code1),
    ("Infinite loop", code2),
    ("Division by zero", code3)
]

for description, code in examples:
    print(f"\nExecuting: {description}")
    result, error = execute_with_timeout(code)
    if error:
        print(f"Error: {error}")
    else:
        print(f"Result: {result}")