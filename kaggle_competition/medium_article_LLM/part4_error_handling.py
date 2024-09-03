import instructor
from openai import OpenAI
from pydantic import BaseModel, Field
from dotenv import load_dotenv, find_dotenv
import os

# Load environment variables from .env
load_dotenv(find_dotenv(filename=".env", usecwd=True, raise_error_if_not_found=True))
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

client = instructor.patch(OpenAI())

class MathSolution(BaseModel):
    answer: str = Field(description="The answer to the math problem")
    python_code: str = Field(description="The Python code that solves the math problem")

def execute_program(code: str):
    try:
        exec(code)
        result = locals().get('answer', None)
        return result, None
    except Exception as e:
        return None, str(e)

def solve_math_problem_validation(problem_text: str):
    try:
        
        solution = client.chat.completions.create(
        model="gpt-4o-mini",  # or whichever model you're using
        response_model=MathSolution,
        messages=[
            {"role": "system", "content": "You are an expert mathematics tutor."},
            {"role": "user", "content": problem_text}
        ],
        max_retries=3
    )
        
        print("Problem solved successfully")
        return solution
    except Exception as e:
        print(f"Failed to solve the problem after 3 retries: {str(e)}")
        return None

def solve_math_problem_execution(problem_text: str, max_retries: int = 3):
    for attempt in range(max_retries):
        try:
            solution = client.chat.completions.create(
                model="gpt-4o-mini",  # or whichever model you're using
                response_model=MathSolution,
                messages=[
                    {"role": "system", "content": "You are an expert mathematics tutor."},
                    {"role": "user", "content": problem_text}
                ],
                max_retries=1
            )
            
            executed_answer, execution_error = execute_program(solution.python_code)
            if execution_error:
                if attempt < max_retries - 1:
                    print(f"Code execution failed: {execution_error}. Retrying...")
                    problem_text += f"\nThe previous code failed with error: {execution_error}. Please provide a corrected version."
                    continue
                else:
                    print("Max retries reached. Unable to solve the problem.")
                    return None
            
            print(f"Problem solved successfully on attempt {attempt + 1}")
            return executed_answer
        
        except Exception as e:
            print(f"Error on attempt {attempt + 1}: {str(e)}")
            if attempt < max_retries - 1:
                print("Retrying...")
            else:
                print("Max retries reached. Unable to solve the problem.")
                return None

# Example usage
if __name__ == "__main__":
    # Example 1: Handling validation errors
    print("Example 1: Handling validation errors")
    problem1 = "What is 2 + 2?"
    solution1 = solve_math_problem_validation(problem1)
    if solution1:
        print(f"Answer: {solution1.answer}")
        print(f"Python code:\n{solution1.python_code}")
    else:
        print("Failed to solve the problem.")

    print("\n" + "="*50 + "\n")

    # Example 2: Handling execution errors
    print("Example 2: Handling execution errors")
    problem2 = "What is the result of 1 divided by 0?"
    answer2 = solve_math_problem_execution(problem2)
    if answer2:
        print(f"Answer: {answer2}")
    else:
        print("Failed to solve the problem.")