import instructor
from pydantic import BaseModel, Field
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv
import sys

load_dotenv(find_dotenv(usecwd=True, raise_error_if_not_found=True))

class MathSolution(BaseModel):
    answer: str = Field(min_length=1)
    python_code: str = Field(min_length=10)

def execute_program(code: str):
    try:
        local_vars = {}
        exec(code, {}, local_vars)
        result = local_vars.get('answer', None)
        return result, None
    except Exception as e:
        return None, str(e)

def get_solution(client, problem_text, max_attempts=2):
    for attempt in range(max_attempts):
        try:
            solution = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are an expert mathematics tutor. Provide Python code to solve the problem and store the result in a variable named 'answer'."},
                    {"role": "user", "content": problem_text}
                ],
                response_model=MathSolution,
                model="gpt-4o-mini",
                max_retries=2
            )

            executed_answer, execution_error = execute_program(solution.python_code)

            if execution_error:
                problem_text = f"The previous code failed to execute with the error: {execution_error}. Please provide a corrected version that solves this problem: {problem_text}"
                continue

            return solution, executed_answer

        except Exception as e:
            print(f"An error occurred on attempt {attempt + 1}: {str(e)}")
            if attempt == max_attempts - 1:
                raise

    raise Exception("Max attempts reached without a valid solution")

def main():
    problem_text = "What is 2 + 2?"
    client = instructor.patch(OpenAI())

    try:
        solution, executed_answer = get_solution(client, problem_text)
        print(f"Problem: {problem_text}")
        print(f"Solution: {solution.answer}")
        print(f"Executed answer: {executed_answer}")
        print(f"Python code:\n{solution.python_code}")
    except Exception as e:
        print(f"Failed to solve the problem: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":