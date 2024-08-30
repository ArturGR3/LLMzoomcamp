import os
import json
from typing import Optional
from pydantic import BaseModel, Field
import instructor
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv

# Load environment variables from .env
load_dotenv(find_dotenv(filename=".env", usecwd=True, raise_error_if_not_found=True))
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Define the MathSolution model
class MathSolution(BaseModel):
    answer: str = Field(..., description="The final numerical answer to the problem")
    step_by_step: str = Field(..., description="A detailed, step-by-step explanation of how to solve the problem")
    python_code: str = Field(..., description="Python code that implements the solution and returns the answer")
    
# Create the prompt function
def create_math_prompt(problem_text: str) -> str:
    return f"""
    Solve the following high school mathematics problem:

    {problem_text}

    Provide your solution in the following format:
    1. The final numerical answer to the problem
    2. A detailed, step-by-step explanation of how to solve the problem
    3. Python code that implements the solution and returns the answer

    Ensure that your Python code is executable and follows these guidelines:
    - Use only Python's built-in functions and the math module
    - Include comments explaining each step
    - Handle potential edge cases or invalid inputs
    - Return the final answer as the last line of the function

    Remember, this is a high school level problem, so advanced mathematical concepts or libraries should not be necessary.
    """

# Initialize the OpenAI client with Instructor
client = instructor.patch(OpenAI())

def solve_math_problem(problem_text: str) -> MathSolution:
    prompt = create_math_prompt(problem_text)
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",  # or whichever model you're using
        response_model=MathSolution,
        messages=[
            {"role": "system", "content": "You are an expert mathematics tutor."},
            {"role": "user", "content": prompt}
        ]
    )
    
    return response

# Example usage
if __name__ == "__main__":
    problem = """
    A rectangle has a length that is 3 units longer than its width. 
    If the perimeter of the rectangle is 26 units, what are the dimensions of the rectangle?
    """
    
    # Show the JSON schema of the MathSolution model
    print(json.dumps(MathSolution.model_json_schema(),indent=4))

    # the output will be 
    ###
    # {
    #     "properties": {
    #         "answer": {
    #             "description": "The final numerical answer to the problem",
    #             "title": "Answer",
    #             "type": "string"
    #         },
    #         "step_by_step": {
    #             "description": "A detailed, step-by-step explanation of how to solve the problem",
    #             "title": "Step By Step",
    #             "type": "string"
    #         },
    #         "python_code": {
    #             "description": "Python code that implements the solution and returns the answer",
    #             "title": "Python Code",
    #             "type": "string"
    #         }
    #     },
    #     "required": [
    #         "answer",
    #         "step_by_step",
    #         "python_code"
    #     ],
    #     "title": "MathSolution",
    #     "type": "object"
    # }
    ### 
    
    solution = solve_math_problem(problem)
    
    # print the raw output of LLM 
    print(json.dumps(solution.model_dump(), indent=4))
    
    # {
    # "answer": "Length = 14 units, Width = 11 units",
    # "step_by_step": "1. Let the width of the rectangle be represented by 'w' units.\n2. According to the problem, the length is 3 units longer than the width, so we can express the length as 'l = w + 3' units.\n3. The formula for the perimeter (P) of a rectangle is given by: P = 2l + 2w.\n4. The problem states that the perimeter is 26 units, so we can set up the equation: 2(w + 3) + 2w = 26.\n5. Simplifying this equation:\n   - 2w + 6 + 2w = 26\n   - 4w + 6 = 26\n   - 4w = 26 - 6\n   - 4w = 20\n   - w = 20 / 4\n   - w = 5 units (this is the width)\n6. Now, substitute w back into the equation for length:\n   - l = w + 3 \n   - l = 5 + 3 = 8 units (this is the length)\n7. Therefore, the dimensions of the rectangle are: Length = 8 units, Width = 5 units.",
    # "python_code": "def find_rectangle_dimensions(perimeter):\n    # Check if the perimeter is even, as the dimensions must be integers\n    if perimeter % 2 != 0:\n        return \"Perimeter must be an even number for integer dimensions.\"\n    # Let the width be 'w'\n    # The formula for perimeter of rectangle: P = 2(l + w)\n    # Since length l = w + 3 (from the problem statement)\n    # We have: 2(w + 3 + w) = perimeter\n    # Simplifying: 2(2w + 3) = perimeter\n    # 4w + 6 = perimeter\n    # Rearranging gives:\n    w = (perimeter - 6) / 4\n    # Calculate the width\n    w = (perimeter - 6) / 4\n    # If width is negative, return an error\n    if w < 0:\n        return \"Invalid dimensions for the given perimeter.\"\n    # Calculate length\n    l = w + 3\n    return (l, w)\n\n# Example usage:\nresult = find_rectangle_dimensions(26)\nresult"
    # }

    print(f"Answer: {solution.answer}")
    print(f"\nStep-by-step solution:\n{solution.step_by_step}")
    print(f"\nPython code:\n{solution.python_code}")
    
    # Answer: Length = 14 units, Width = 11 units

    # Step-by-step solution:
    # 1. Let the width of the rectangle be represented by 'w'.  
    # 2. According to the problem, the length 'l' is 3 units longer than the width: l = w + 3.  
    # 3. The formula for the perimeter 'P' of a rectangle is given by: P = 2l + 2w.  
    # 4. Substitute the given perimeter into the formula: 2l + 2w = 26.  
    # 5. Replace 'l' with 'w + 3': 2(w + 3) + 2w = 26.  
    # 6. Simplify the equation: 2w + 6 + 2w = 26.  
    # 7. Combine like terms: 4w + 6 = 26.  
    # 8. Subtract 6 from both sides: 4w = 20.  
    # 9. Divide by 4: w = 5.  
    # 10. Now, find the length using l = w + 3: l = 5 + 3 = 8.  
    # 11. So the dimensions of the rectangle are: Width = 5 units and Length = 8 units.

    # Python code:
    # def rectangle_dimensions(perimeter):
    #     # Given perimeter of the rectangle
    #     P = perimeter
    #     # The variable for width
    #     w = 0
    #     # Loop to find the width and corresponding length
    #     for w in range(1, P//2):  # Width should be at least 1 and less than half the perimeter
    #         l = w + 3  # Length is 3 units longer than the width
    #         if 2 * (l + w) == P:  # Check if perimeter matches
    #             # Return dimensions when found
    #             return (l, w)
    #     # In case no valid dimensions found
    #     return None

    # # Example usage
    # result = rectangle_dimensions(26)
    # # Return the final answer as Length and Width
    # result

