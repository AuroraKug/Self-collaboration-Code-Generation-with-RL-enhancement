import os
import openai
import time
import copy
import json
import argparse
import tqdm
import re

from core import interface
from utils import code_truncate, construct_system_message
from roles.instruction import INSTRUCTPLAN, INSTRUCTREPORT, INSTRUCTCODE

class Coder(object):
    def __init__(self, TEAM, PYTHON_DEVELOPER, requirement, model='gpt-3.5-turbo-0301', majority=1, max_tokens=512,
                                temperature=0.0, top_p=1.0):
        self.model = model
        self.majority = majority
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.history_message = []
        self.requirement = requirement

        self.itf = interface.ProgramInterface(
            stop='',
            verbose=False,
            model = self.model,
        )

        system_message = construct_system_message(requirement, PYTHON_DEVELOPER, TEAM)

        self.history_message_append(system_message)

    def implement(self, report, is_init=False):
        self.construct_with_report(report, is_init)
        
        naive_code_candidates = []
        try:
            # responses will be a list of strings if self.majority > 1
            responses = self.itf.run(prompt=self.history_message, majority_at = self.majority, max_tokens=self.max_tokens, temperature=self.temperature, top_p=self.top_p)
        except Exception as e:
            print(e)
            print("implement fail")
            time.sleep(5)
            # Return an empty list or a list with an error marker if preferred
            return ["error"] # Or simply return [] and let Session handle it

        # Remove the user instruction message that was added by construct_with_report
        # This prepares history for the next potential call or for Session to append the chosen assistant message.
        if self.history_message and self.history_message[-1]["role"] == "user" and "INSTRUCTCODE" in self.history_message[-1]["content"]:
            self.history_message.pop()
            if self.history_message and self.history_message[-1]["role"] == "user" and ("INSTRUCTPLAN" in self.history_message[-1]["content"] or "INSTRUCTREPORT" in self.history_message[-1]["content"]):
                 self.history_message.pop()


        for response_text in responses:
            if 'gpt' not in self.model:
                # Attempt to handle the non-GPT model response format if necessary
                # This part might need adjustment based on actual non-GPT model output for multiple completions
                generation = response_text[response_text.find("def"):] if "def" in response_text else response_text
                tem = [s for s in generation.split('\\n\\n') if 'def ' in s or s.startswith(' ') or 'class ' in s]
                code = '\\n\\n'.join(tem).strip('```').strip()
            else:
                code = code_truncate(response_text)
            
            if code.strip(): # Add only non-empty code candidates
                naive_code_candidates.append(code)
        
        if not naive_code_candidates: # If all responses resulted in empty code
            return ["error"]

        return naive_code_candidates
    
    def history_message_append(self, system_message, role="user"):
        self.history_message.append({
            "role": role,
            "content": system_message
        })
        
    def construct_with_report(self, report, is_init=False):
        if report != "":
            if is_init:
                instruction = INSTRUCTPLAN.format(report=report.strip())
            else:
                instruction = INSTRUCTREPORT.format(report=report.strip())
            self.history_message_append(instruction)
            self.history_message_append(INSTRUCTCODE.format(requirement=self.requirement))

    reflection_prompt_template = """
You are an expert Python code reviewer and corrector. You are given a programming requirement,
the previous version of the code (if any), the feedback from the last round (if any),
and the current draft of the Python code.
Your task is to review the current draft and improve it before it's sent for testing.

Requirement:
{requirement}

Previous Code (if any):
```python
{previous_code}
```

Feedback from Last Round (e.g., test results or analyst plan):
{history_feedback}

Current Draft Code:
```python
{current_code}
```

Review Checklist & Correction Guidelines:
1.  Correctness: Does the code seem to logically address the requirement? Are there any obvious bugs or logical flaws?
2.  Completeness: Does the code handle edge cases mentioned or implied in the requirement? (e.g., empty inputs, invalid inputs)
3.  Clarity & Readability: Is the code easy to understand? Are variable names meaningful?
4.  Efficiency: Are there any obvious inefficient patterns that could be improved without major refactoring? (Avoid premature optimization unless clearly problematic)
5.  Error Handling: Does the code include basic error handling if appropriate for the requirement?
6.  Compliance with Feedback: If `history_feedback` indicates previous errors, has the current code addressed them?
7.  Function Definition: Ensure there is one clearly defined Python function that is the main entry point and its name is easily identifiable.

Based on your review, provide an improved version of the code.
If you believe the current code is already good and needs no changes, output the original code.
If you identify issues but cannot fix them, you can try to explain the issue and return the original code, or make a best-effort attempt.
Output only the Python code block for the improved function. Do not include any other text or explanation outside the code block.

Improved Code:
```python
[Insert your improved Python code here]
```
"""

    def self_reflect_and_correct(self, requirement, current_code, previous_code, history_feedback):
        if not current_code or current_code.strip() == "" or current_code == "error":
            return current_code

        reflection_system_message = {"role": "system", "content": "You are an expert Python code reviewer and corrector."}
        
        reflection_user_prompt_content = self.reflection_prompt_template.format(
            requirement=requirement,
            previous_code=previous_code if previous_code and previous_code.strip() else "# No previous code provided for this reflection.",
            history_feedback=history_feedback if history_feedback and history_feedback.strip() else "# No specific feedback from the last round, or this is an initial plan.",
            current_code=current_code
        )
        reflection_prompt_messages = [
            reflection_system_message,
            {"role": "user", "content": reflection_user_prompt_content}
        ]

        try:
            responses = self.itf.run(
                prompt=reflection_prompt_messages, 
                majority_at=1, 
                max_tokens=self.max_tokens, 
                temperature=0.0,   
                top_p=self.top_p 
            )
            reflected_code_raw = responses[0] 
            reflected_code = self._extract_code_from_response(reflected_code_raw)

        except Exception as e:
            print(f"Error during Coder self-reflection: {e}")
            return current_code 
        
        if not reflected_code or reflected_code.strip() == "":
            return current_code 
        return reflected_code

    def _extract_code_from_response(self, response_text):
        if not response_text:
            return ""
        
        match = re.search(r"```python\n(.*?)\n```", response_text, re.DOTALL)
        if match:
            return match.group(1).strip()
        
        return code_truncate(response_text)
