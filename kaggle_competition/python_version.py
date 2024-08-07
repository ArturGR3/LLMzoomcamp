import pandas as pd
from openai import OpenAI
from pydantic import BaseModel, Field, field_validator
import instructor
from textwrap import dedent
from typing import Literal
import traceback
from kaggle_client import KaggleClient
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from instructor import llm_validator

# Download the data parametert
download_data = False
display_data = False 


# # load openai api key
# load_dotenv(find_dotenv(filename=".env", usecwd=True, raise_error_if_not_found=True))
# os.environ["OPENAI_API_KEY"]=os.getenv("OPENAI_API_KEY")

if download_data:
    llm_competition = KaggleClient("llm-zoomcamp-2024-competition", "answer")
    llm_competition.download_data()

client = instructor.patch(OpenAI())

prefix = """
# Answer this question by implementing a solver()
# function.
def solver():
    # The plan for our solution is following:
    # 1.
""".strip()

def execute_program(code: str):
    # Strip any leading or trailing whitespace from the code and append a new line and "ans = solver()" to it
    code = code.strip() + "\nans = solver()"
    # Print the code that is going to be executed
    print("Executing code:\n", code)
    try:
        # Create an empty dictionary to serve as the global namespace for executing the code
        exec_globals = {}
        # Execute the code within the exec_globals namespace
        exec(code, exec_globals)
        # Return the value of the "ans" variable from the exec_globals namespace
        return exec_globals.get("ans")
    except Exception as e:
        # Print any error that occurred during code execution
        print(f"Error executing code: {str(e)}")
        print(traceback.format_exc())
        # Return None if an error occurred
        return None

class ProgramExecution(BaseModel):
    program_code: str = Field(description="Program Code that once executed contains the final answer")

    @field_validator("program_code")
    @classmethod
    def ensure_valid_code(cls, v: str) -> str:
        if not v.startswith(prefix):
            raise ValueError(f"Program Code must begin with the desired prefix of {prefix}")

        answer = execute_program(v)
        if answer is None:
            raise ValueError("Error occurred during program execution or no answer was returned")

        return str(answer)

def generate_intermediate_reasoning(query: str, max_retries: int = 3):
    attempt = 0
    error_message = ""
    
    while attempt < max_retries:
        print(f"Attempt {attempt + 1}/{max_retries}")
        try:
            # Modify the prompt to include the error message if any
            prompt = dedent(
                f"""
                You are a world class AI system that excels
                at answering user queries in a systematic
                and detailed manner. You are about to be
                passed a user query to respond to. Make sure to
                generate a valid Python program that can be
                executed to answer the user query. Provide step-by-step reasoning
                as comments in the code.

                Your response MUST begin with EXACTLY the following prefix:

                {prefix}

                After this prefix, continue with your implementation of the solver() function.
                Your program must define and call a solver() function
                that returns the final answer as a number.

                {error_message}
                """
            )
            
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                temperature=0,  # Lower temperature for more deterministic output
                max_retries=2,
                messages=[
                    {
                        "role": "system",
                        "content": prompt,
                    },
                    {
                        "role": "user",
                        "content": query,
                    },
                ],
                response_model=ProgramExecution,
            )
            print("Raw LLM output:", response.program_code)
            return response
        except instructor.exceptions.InstructorRetryException as e:
            print(f"InstructorRetryException: {str(e)}")
            print("Error type:", type(e).__name__)
            print(traceback.format_exc())
            
            # Try to extract and print the raw LLM output if available
            try:
                if hasattr(e, 'response') and hasattr(e.response, 'choices'):
                    raw_output = e.response.choices[0].message.content
                    print("Raw LLM output:", raw_output)
                    if not raw_output.startswith(prefix):
                        print("Does output start with prefix?", raw_output.startswith(prefix))
                        print("First 100 characters of output:", raw_output[:100])
                        print("First 100 characters of expected prefix:", prefix[:100])
                        error_message = f"Previous attempt failed with error: {str(e)}"
                        attempt += 1
                        continue
            except:
                print("Unable to extract raw LLM output")
            
            return None
        except Exception as e:
            print(f"Error in generate_intermediate_reasoning: {str(e)}")
            print("Error type:", type(e).__name__)
            print(traceback.format_exc())
            return None
    print("Max retries reached. Unable to generate valid code.")
    return None


# Example usage
query = "In a company of 30 people, 25 use the social network \"Odnoklassniki\" and 10 use the social network \"VKontakte\". Choose the statements that are true under the given conditions.\n\n\\begin{center}\n\\begin{tabularx}{\\textwidth}{p{0.1cm}X}  \n1) & In this company, there will be 10 people who do not use either \"Odnoklassniki\" or \"VKontakte\". \\\\ \n2) & In this company, there will be at least 5 people using both networks. \\\\ \n3) & There will not be a single person in this company who uses only \"Odnoklassniki\". \\\\ \n4) & No more than 10 people in this company use both networks. \\\\ \n\\end{tabularx}\\end{center}\n\nIn the answer, write the numbers of the selected statements without spaces, commas, or other additional symbols."
reasoning = generate_intermediate_reasoning(query)
answer = reasoning.program_code

train = pd.read_csv("data/train.csv")
train.iloc[0].to_dict()


def process_data(row_json):
    # Iterate over the dataset with tqdm progress bar
    problem_id = row_json['problem_id']
    problem_text = row_json['problem_text']
    ground_truth = row_json['answer']
    answer_llm = generate_intermediate_reasoning(problem_text)
  
    return    {
        "problem_id": problem_id,
        "problem_text": problem_text,
        "ground_truth": ground_truth,
        "answer_llm": answer_llm    
    }


pool = ThreadPoolExecutor(max_workers=4)

def map_progress(pool, seq, f):
    results = []

    with tqdm(total=len(seq)) as progress:
        futures = []

        for el in seq:
            future = pool.submit(f, el)
            future.add_done_callback(lambda p: progress.update())
            futures.append(future)

        for future in futures:
            result = future.result()
            results.append(result)
    return results

results = map_progress(pool, train.iloc[:10].to_dict(orient="records"), process_data)

# how to select a key from a dictionary results
resultts_pd = pd.DataFrame(results)


