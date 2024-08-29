import os
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
    
    solution = solve_math_problem(problem)

    print(f"Answer: {solution.answer}")
    print(f"\nStep-by-step solution:\n{solution.step_by_step}")
    print(f"\nPython code:\n{solution.python_code}")

