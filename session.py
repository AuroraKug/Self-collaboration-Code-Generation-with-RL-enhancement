from roles import Analyst, Coder, Tester
from utils import find_method_name
import time
from utils import code_truncate


class Session(object):
    def __init__(self, TEAM, ANALYST, PYTHON_DEVELOPER, TESTER, requirement, model='gpt-3.5-turbo-0301', majority=1, max_tokens=512,
                                temperature=0.0, top_p=1.0, max_round=4, before_func=''):

        self.session_history = {}
        self.max_round = max_round
        self.before_func = before_func
        self.requirement = requirement
        self.analyst = Analyst(TEAM, ANALYST, requirement, model, majority, max_tokens, temperature, top_p)
        self.coder = Coder(TEAM, PYTHON_DEVELOPER, requirement, model, majority, max_tokens, temperature, top_p)
        self.tester = Tester(TEAM, TESTER, requirement, model, majority, max_tokens, temperature, top_p)
    
    def run_session(self):
        plan = self.analyst.analyze()
        current_report_for_coder = plan # Renamed 'report' to avoid confusion in the loop
        is_init=True
        self.session_history["plan"] = plan
        final_code_of_the_round = "" # Renamed 'code' to avoid confusion
        previous_code_for_reflection = ""

        for i in range(self.max_round):
            # Coder generates a list of naive code candidates
            naive_code_candidates = self.coder.implement(current_report_for_coder, is_init)

            best_code_candidate_this_round = ""
            # Initialize with a generic failure, to be updated by the first processed candidate or a passing one
            best_answer_report_this_round = "error - no valid candidates processed"
            chosen_naive_code_for_history = ""
            chosen_reflected_code_for_history = ""
            
            # Handle cases where Coder implement returns an error or empty list directly
            if not naive_code_candidates or naive_code_candidates == ["error"]:
                if i == 0:
                    final_code_of_the_round = "error"
                else:
                    final_code_of_the_round = self.session_history.get(f'Round_{i-1}', {}).get("code", "error")
                break # Break from max_round loop

            for naive_candidate in naive_code_candidates:
                if not naive_candidate or naive_candidate.strip() == "" or naive_candidate == "error":
                    continue

                reflected_candidate = self.coder.self_reflect_and_correct(
                    requirement=self.requirement,
                    current_code=naive_candidate,
                    previous_code=previous_code_for_reflection,
                    history_feedback=current_report_for_coder # Feedback from previous round or plan
                )

                current_code_to_test = reflected_candidate
                if not current_code_to_test or current_code_to_test.strip() == "" or current_code_to_test == "error":
                    current_code_to_test = naive_candidate # Fallback to naive if reflection fails
                
                if not current_code_to_test or current_code_to_test.strip() == "" or current_code_to_test == "error":
                    continue # Skip this candidate if still bad

                method_name = find_method_name(current_code_to_test)
                if not method_name:
                    # If this is the first candidate being processed and it has no method name,
                    # capture its state for potential use if no other candidate works out.
                    if best_code_candidate_this_round == "":
                        best_code_candidate_this_round = current_code_to_test # Or an error marker
                        best_answer_report_this_round = "error - no method name found in candidate"
                        chosen_naive_code_for_history = naive_candidate
                        chosen_reflected_code_for_history = reflected_candidate
                    continue # Try next candidate

                tests = self.tester.test(current_code_to_test)
                test_report_str = code_truncate(tests)
                
                current_answer_report = unsafe_execute(
                    self.before_func + current_code_to_test + '\n' + test_report_str + '\n' + f'check({method_name})',
                    ''
                )

                # If this is the first valid candidate being fully processed, its result becomes the current best
                if best_code_candidate_this_round == "" or (best_answer_report_this_round != "Code Test Passed." and current_answer_report == "Code Test Passed.") :
                    best_code_candidate_this_round = current_code_to_test
                    best_answer_report_this_round = current_answer_report
                    chosen_naive_code_for_history = naive_candidate
                    chosen_reflected_code_for_history = reflected_candidate

                if current_answer_report == "Code Test Passed.":
                    break # Found a passing candidate, no need to check others in this round
            
            final_code_of_the_round = best_code_candidate_this_round

            # If after checking all candidates, final_code_of_the_round is empty (e.g. all candidates were invalid before testing)
            if not final_code_of_the_round.strip():
                if i == 0:
                    final_code_of_the_round = "error"
                else:
                    final_code_of_the_round = self.session_history.get(f'Round_{i-1}', {}).get("code", "error")
                # Update Coder history with this error status to prevent issues if it expects an assistant message
                if final_code_of_the_round == "error":
                    self.coder.history_message_append("error: no valid code generated", "assistant")
                break
            
            if final_code_of_the_round == "error": # Propagated from previous checks
                self.coder.history_message_append("error: no valid code generated", "assistant")
                break

            # Add the chosen code to Coder's history for the next round
            self.coder.history_message_append(final_code_of_the_round, "assistant")
            previous_code_for_reflection = final_code_of_the_round

            # Prepare report for the next round (or for session history if last round)
            current_report_for_coder = f'The compilation output of the preceding code is: {best_answer_report_this_round}'

            self.session_history[f'Round_{i}'] = {
                "code": final_code_of_the_round,
                "report": current_report_for_coder,
                "chosen_naive_code": chosen_naive_code_for_history,
                "chosen_reflected_code": chosen_reflected_code_for_history,
                "answer_report_of_chosen": best_answer_report_this_round
            }

            if i == self.max_round - 1: # Last round
                break

            if best_answer_report_this_round == "Code Test Passed.":
                break
            
            # If the best answer report indicates a failure that should stop iteration (e.g. not a simple test fail)
            # This part of logic might need to be more refined based on how `best_answer_report_this_round` signals critical errors
            if "error" in best_answer_report_this_round.lower() and best_answer_report_this_round != "error - no valid candidates processed" and best_answer_report_this_round != "error - no method name found in candidate":
                # Potentially break if it is a critical error beyond just test failures
                # For now, we continue to allow iterative fixing unless it's a pass.
                pass 

            is_init = False

        self.analyst.itf.clear_history()
        self.coder.itf.clear_history()
        self.tester.itf.clear_history()

        return final_code_of_the_round, self.session_history

    def run_analyst_coder(self):
        plan = self.analyst.analyze()
        is_init=True
        self.session_history["plan"] = plan
        code = self.coder.implement(plan, is_init)

        if (plan == "error") or (code == "error"):
            code = "error"

        self.analyst.itf.clear_history()
        self.coder.itf.clear_history()
        self.tester.itf.clear_history()

        return code, self.session_history


    def run_coder_tester(self):
        report = ""
        is_init=True
        code = ""
        
        for i in range(self.max_round):

            naivecode = self.coder.implement(report, is_init)
            if find_method_name(naivecode):
                code = naivecode

            if code.strip() == "":
                if i == 0:
                    code = self.coder.implement(report, is_init=True)
                else:
                    code = self.session_history['Round_{}'.format(i-1)]["code"]
                break
            
            if i == self.max_round-1:
                self.session_history['Round_{}'.format(i)] = {"code": code}
                break
            tests = self.tester.test(code)
            test_report = code_truncate(tests)
            answer_report = unsafe_execute(self.before_func+code+'\n'+test_report+'\n'+f'check({method_name})', '')
            report = f'The compilation output of the preceding code is: {answer_report}'

            is_init = False
            self.session_history['Round_{}'.format(i)] = {"code": code, "report": report}

            if (code == "error") or (report == "error"):
                code = "error"
                break
            
            if report == "Code Test Passed.":
                break

        self.analyst.itf.clear_history()
        self.coder.itf.clear_history()
        self.tester.itf.clear_history()

        return code, self.session_history

    def run_coder_only(self):
        plan = ""
        code = self.coder.implement(plan, is_init=True)
        self.coder.itf.clear_history()
        return code, self.session_history


import contextlib
import faulthandler
import io
import os
import platform
import signal
import tempfile 

def unsafe_execute(code, report):

        with create_tempdir():

            # These system calls are needed when cleaning up tempdir.
            import os
            import shutil
            rmtree = shutil.rmtree
            rmdir = os.rmdir
            chdir = os.chdir 

            # Disable functionalities that can make destructive changes to the test.
            reliability_guard()

            # Construct the check program and run it.
            check_program = (
                code + report
            )

            try:
                exec_globals = {}
                with swallow_io():
                    timeout = 10
                    with time_limit(timeout):
                        exec(check_program, exec_globals)
                result = "Code Test Passed."
            except AssertionError as e:
                result = f"failed with AssertionError. {e}"
            except TimeoutException:
                result = "timed out"
            except BaseException as e:
                result = f"{e}"


            # Needed for cleaning up.
            shutil.rmtree = rmtree
            os.rmdir = rmdir
            os.chdir = chdir
            return result


def reliability_guard(maximum_memory_bytes = None):
    """
    This disables various destructive functions and prevents the generated code
    from interfering with the test (e.g. fork bomb, killing other processes,
    removing filesystem files, etc.)

    WARNING
    This function is NOT a security sandbox. Untrusted code, including, model-
    generated code, should not be blindly executed outside of one. See the 
    Codex paper for more information about OpenAI's code sandbox, and proceed
    with caution.
    """

    if maximum_memory_bytes is not None:
        import resource
        resource.setrlimit(resource.RLIMIT_AS, (maximum_memory_bytes, maximum_memory_bytes))
        resource.setrlimit(resource.RLIMIT_DATA, (maximum_memory_bytes, maximum_memory_bytes))
        if not platform.uname().system == 'Darwin':
            resource.setrlimit(resource.RLIMIT_STACK, (maximum_memory_bytes, maximum_memory_bytes))

    faulthandler.disable()

    import builtins
    builtins.exit = None
    builtins.quit = None

    import os
    os.environ['OMP_NUM_THREADS'] = '1'

    os.rmdir = None
    os.chdir = None

    import shutil
    shutil.rmtree = None
    shutil.move = None
    shutil.chown = None

    import subprocess
    subprocess.Popen = None  # type: ignore

    __builtins__['help'] = None

    import sys
    sys.modules['ipdb'] = None
    sys.modules['joblib'] = None
    sys.modules['resource'] = None
    sys.modules['psutil'] = None
    sys.modules['tkinter'] = None
    
@contextlib.contextmanager
def time_limit(seconds: float):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")
    signal.setitimer(signal.ITIMER_REAL, seconds)
    signal.signal(signal.SIGALRM, signal_handler)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


@contextlib.contextmanager
def swallow_io():
    stream = WriteOnlyStringIO()
    with contextlib.redirect_stdout(stream):
        with contextlib.redirect_stderr(stream):
            with redirect_stdin(stream):
                yield


@contextlib.contextmanager
def create_tempdir():
    with tempfile.TemporaryDirectory() as dirname:
        with chdir(dirname):
            yield dirname
            
class TimeoutException(Exception):
    pass


class WriteOnlyStringIO(io.StringIO):
    """ StringIO that throws an exception when it's read from """

    def read(self, *args, **kwargs):
        raise IOError

    def readline(self, *args, **kwargs):
        raise IOError

    def readlines(self, *args, **kwargs):
        raise IOError

    def readable(self, *args, **kwargs):
        """ Returns True if the IO object can be read. """
        return False


class redirect_stdin(contextlib._RedirectStream):  # type: ignore
    _stream = 'stdin'


@contextlib.contextmanager
def chdir(root):
    if root == ".":
        yield
        return
    cwd = os.getcwd()
    os.chdir(root)
    try:
        yield
    except BaseException as exc:
        raise exc
    finally:
        os.chdir(cwd)