import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pydantic import BaseModel, Field, ValidationError
from typing import List, Union

# Simulating LLM API
def llm_api_call(problem: str) -> dict:
    time.sleep(1)  # Simulate API delay
    if problem == "5 / 0":
        return {"error": "Division by zero"}
    elif problem == "sqrt(-1)":
        return {"answer": "i", "explanation": "The square root of -1 is i"}
    else:
        try:
            answer = eval(problem)
            return {"answer": answer, "explanation": f"The result of {problem} is {answer}"}
        except:
            return {"error": "Invalid expression"}

# Pydantic model for structured output
class MathSolution(BaseModel):
    problem: str = Field(..., description="The original math problem")
    answer: float = Field(..., description="The numerical answer to the problem")
    explanation: str = Field(..., description="Explanation of the solution")

def solve_problem(problem: str) -> Union[MathSolution, str]:
    try:
        result = llm_api_call(problem)
        if "error" in result:
            return f"Error: {result['error']}"
        
        return MathSolution(
            problem=problem,
            answer=float(result['answer']),
            explanation=result['explanation']
        )
    except ValidationError as e:
        return f"Validation Error: {str(e)}"
    except Exception as e:
        return f"Unexpected Error: {str(e)}"

def process_models_threaded(problems: List[str], max_workers: int = 3) -> List[Union[MathSolution, str]]:
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(solve_problem, problem) for problem in problems]
        for future in as_completed(futures):
            results.append(future.result())
    return results

if __name__ == "__main__":
    math_problems = [
        "2 + 2",
        "3 * 4 - 2",
        "5 / 0",  # This will cause an error
        "10 + 5",
        "sqrt(-1)",  # This will cause a validation error
        "7 * 8"
    ]

    start_time = time.time()
    results = process_models_threaded(math_problems)
    end_time = time.time()

    for result in results:
        if isinstance(result, MathSolution):
            print(f"Problem: {result.problem}")
            print(f"Answer: {result.answer}")
            print(f"Explanation: {result.explanation}")
        else:
            print(result)
        print()

    print(f"Total time taken: {end_time - start_time:.2f} seconds")
    
    # Problem: 2 + 2
    # Answer: 4.0
    # Explanation: The result of 2 + 2 is 4

    # Error: Division by zero

    # Problem: 3 * 4 - 2
    # Answer: 10.0
    # Explanation: The result of 3 * 4 - 2 is 10

    # Problem: 10 + 5
    # Answer: 15.0
    # Explanation: The result of 10 + 5 is 15

    # Problem: 7 * 8
    # Answer: 56.0
    # Explanation: The result of 7 * 8 is 56

    # Unexpected Error: could not convert string to float: 'i'