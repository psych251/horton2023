import json
import random
import ast
import re
import sys
sys.path.append('../')

from utils.settings import * 
from utils.global_methods import *
from generative_agent.prompt_template.gpt_structure import *
from generative_agent.prompt_template.print_prompt import *


prompt_dir = "generative_agent/prompt_template/prompts"


##############################################################################
#######                         HELPER FUNCTIONS                       #######
##############################################################################

def extract_first_json_dict(input_str):
  try:
    # Replace curly quotes with standard double quotes
    input_str = (input_str.replace("“", "\"")
                          .replace("”", "\"")
                          .replace("‘", "'")
                          .replace("’", "'"))
    
    # Find the first occurrence of '{' in the input_str
    start_index = input_str.index('{')
    
    # Initialize a count to keep track of open and close braces
    count = 1
    end_index = start_index + 1
    
    # Loop to find the closing '}' for the first JSON dictionary
    while count > 0 and end_index < len(input_str):
        if input_str[end_index] == '{':
            count += 1
        elif input_str[end_index] == '}':
            count -= 1
        end_index += 1
    
    # Extract the JSON substring
    json_str = input_str[start_index:end_index]
    
    # Parse the JSON string into a Python dictionary
    json_dict = json.loads(json_str)
    
    return json_dict
  except ValueError:
    # Handle the case where the JSON parsing fails
    return None


##############################################################################
###                         GENERATIVE AGENT CLASS                         ###
##############################################################################

def run_gpt_generate_batch_importance(name, records, 
      test_input=None, verbose=False): 
  def create_prompt_input(name, records, test_input=None):
    record_count = str(len(records))
    records_str = ""
    for count, r in enumerate(records): 
      records_str += f"Item {str(count+1)}:\n"
      records_str += f"{r}\n"

    prompt_input = [record_count, records_str, name]
    return prompt_input

  def __chat_func_clean_up(gpt_response, prompt=""): 
    gpt_response = json.loads(gpt_response) 
    return gpt_response

  def __chat_func_validate(gpt_response, prompt=""): 
    try: 
      response = __chat_func_clean_up(gpt_response)
      return True
    except:
      return False 

  def get_fail_safe():
    return None

  prompt_template = f"{prompt_dir}/batch_importance_v1.txt" 
  prompt_input = create_prompt_input(name, records) 
  prompt = generate_prompt(prompt_input, prompt_template)
  fail_safe = get_fail_safe() 
  output = chat_safe_generate(prompt, "GPT4", 5, fail_safe,
             __chat_func_validate, __chat_func_clean_up, verbose)

  if DEBUG or verbose: 
    print_run_prompts(prompt_template, prompt_input, prompt, output)
  return output, [output, prompt, prompt_input, fail_safe]


def run_gpt_generate_profiler_reflections_v2(name, observation_str, 
      test_input=None, verbose=False): 
  def create_prompt_input(name, observation_str, test_input=None):
    prompt_input = [observation_str, name, name, name, name, name]
    return prompt_input

  def __chat_func_clean_up(gpt_response, prompt=""): 
    gpt_response = list(extract_first_json_dict(gpt_response).values())
    return gpt_response

  def __chat_func_validate(gpt_response, prompt=""): 
    try: 
      response = __chat_func_clean_up(gpt_response)
      return True
    except:
      return False 

  def get_fail_safe():
    return None

  prompt_template = f"{prompt_dir}/profiler_reflection_v2.txt" 
  prompt_input = create_prompt_input(name, observation_str) 
  prompt = generate_prompt(prompt_input, prompt_template)
  fail_safe = get_fail_safe() 
  output = chat_safe_generate(prompt, "ChatGPT", 5, fail_safe,
             __chat_func_validate, __chat_func_clean_up, verbose)

  if DEBUG or verbose: 
    print_run_prompts(prompt_template, prompt_input, prompt, output)
  return output, [output, prompt, prompt_input, fail_safe]


def run_gpt_generate_reflections(observation_str, 
      test_input=None, verbose=False):
  def create_prompt_input(observation_str, test_input=None):
    prompt_input = [observation_str]
    return prompt_input

  def __chat_func_clean_up(gpt_response, prompt=""): 
    gpt_response = list(extract_first_json_dict(gpt_response)["output"])
    return gpt_response

  def __chat_func_validate(gpt_response, prompt=""): 
    try: 
      response = __chat_func_clean_up(gpt_response)
      for i in response: 
        if type(i) != type("a"): 
          return False
      return True
    except:
      return False 

  def get_fail_safe():
    return None

  prompt_template = f"{prompt_dir}/reflection_v1.txt" 
  prompt_input = create_prompt_input(observation_str) 
  prompt = generate_prompt(prompt_input, prompt_template)
  fail_safe = get_fail_safe() 
  output = chat_safe_generate(prompt, "ChatGPT", 5, fail_safe,
             __chat_func_validate, __chat_func_clean_up, verbose)

  if DEBUG or verbose: 
    print_run_prompts(prompt_template, prompt_input, prompt, output)
  return output, [output, prompt, prompt_input, fail_safe]



##############################################################################
###                         AGENT INTERFACE CLASS                          ###
##############################################################################

def run_gpt_generate_corr_questions(inquiry, test_input=None, verbose=False):
  def create_prompt_input(inquiry, test_input=None):
    prompt_input = [inquiry]
    return prompt_input

  def __chat_func_clean_up(gpt_response, prompt=""): 
    gpt_response = extract_first_json_dict(
                     gpt_response)["correlated questions"]
    return gpt_response

  def __chat_func_validate(gpt_response, prompt=""): 
    try: 
      response = __chat_func_clean_up(gpt_response)
    except:
      return False 

  def get_fail_safe():
    return None

  prompt_template = f"{prompt_dir}/corr_inquiry_v1.txt" 
  prompt_input = create_prompt_input(inquiry) 
  prompt = generate_prompt(prompt_input, prompt_template)
  fail_safe = get_fail_safe() 
  output = chat_safe_generate(prompt, "ChatGPT", 5, fail_safe,
                        __chat_func_validate, __chat_func_clean_up, verbose)

  if DEBUG or verbose: 
    print_run_prompts(prompt_template, prompt_input, prompt, output)
  return output, [output, prompt, prompt_input, fail_safe]


def run_gpt_generate_inquiry_response_hybrid(name, retrieved_obs_str,
      retrieved_ref_str, inquiry, test_input=None, verbose=False):
  def create_reason_prompt_input(name, retrieved_obs_str, retrieved_ref_str, 
        inquiry, test_input=None):
    prompt_input = [name, name, name, name, 
                    retrieved_obs_str, 
                    name, name, 
                    retrieved_ref_str, 
                    name, name,
                    inquiry, 
                    name, name, name, name, name, name,
                    inquiry]
    return prompt_input
  
  def create_prompt_input(name, inquiry, reason_output, test_input=None):
    prompt_input = [name, reason_output, inquiry, name]
    return prompt_input

  def __chat_func_clean_up_reason(gpt_response, prompt=""): 
    gpt_response = json.loads(gpt_response)
    return gpt_response
  
  def __chat_func_clean_up(gpt_response, prompt=""): 
    return gpt_response

  def __chat_func_validate_reason(gpt_response, prompt=""): 
    try: 
      fields = ["question 0"]
      response = json.loads(gpt_response)
      for field in fields: 
        if field not in response: 
          return False
      return True
    except:
      return False 
    
  def __chat_func_validate(gpt_response, prompt=""): 
    if gpt_response != None:
      return True
    else:
      return False


  def get_fail_safe():
    return None

  prompt_template_reason = f"{prompt_dir}/inquiry_reason_v1.txt" 
  prompt_input_reason = create_reason_prompt_input(name, retrieved_obs_str, retrieved_ref_str, inquiry)
  prompt_reason = generate_prompt(prompt_input_reason, prompt_template_reason)
  fail_safe = get_fail_safe() 
  reason_output = chat_safe_generate(prompt_reason, "ChatGPT", 5, fail_safe,
                        __chat_func_validate_reason, __chat_func_clean_up_reason, verbose)["question 0"]
  
  prompt_template = f"{prompt_dir}/inquiry_response_hybrid_v1.txt"
  prompt_input = create_prompt_input(name, inquiry, reason_output)
  prompt = generate_prompt(prompt_input, prompt_template)
  fail_safe = get_fail_safe()
  output = nonchat_safe_generate(prompt, 100, "gpt-3.5-turbo-instruct", 5, fail_safe,
                        __chat_func_validate, __chat_func_clean_up, verbose)

  if DEBUG or verbose: 
    print_run_prompts(prompt_template, prompt_input, prompt, output)
  return output, [output, prompt, prompt_input, fail_safe]







def run_gpt_generate_corr_topics(inquiry, test_input=None, verbose=False):
  def create_prompt_input(inquiry, test_input=None):
    prompt_input = [inquiry]
    return prompt_input

  def __chat_func_clean_up(gpt_response, prompt=""): 
    gpt_response = extract_first_json_dict(gpt_response)
    gpt_response = list(gpt_response.values())
    return gpt_response

  def __chat_func_validate(gpt_response, prompt=""): 
    if verbose:
      print (gpt_response)
    try: 
      fields = ["point 1", "point 2", "point 3"]
      gpt_response = extract_first_json_dict(gpt_response)
      for field in fields: 
        if field not in gpt_response: 
          return False
      return True
    except:
      return False 

  def get_fail_safe():
    return None

  prompt_template = f"{prompt_dir}/corr_topics_v1.txt" 
  prompt_input = create_prompt_input(inquiry) 
  prompt = generate_prompt(prompt_input, prompt_template)
  fail_safe = get_fail_safe() 
  output = chat_safe_generate(prompt, "ChatGPT", 5, fail_safe,
                        __chat_func_validate, __chat_func_clean_up, verbose)
  if DEBUG or verbose: 
    print_run_prompts(prompt_template, prompt_input, prompt, output)
  return output, [output, prompt, prompt_input, fail_safe]


def run_gpt_generate_corr_topics_HyDE(name, inquiry, curr_ret_str, test_input=None, verbose=False):
  def create_prompt_input(name, inquiry, curr_ret_str, test_input=None):
    prompt_input = [name, curr_ret_str, name, inquiry, name]
    return prompt_input

  def __chat_func_clean_up(gpt_response, prompt=""): 
    gpt_response = extract_first_json_dict(gpt_response)
    gpt_response = list(gpt_response.values())
    return gpt_response

  def __chat_func_validate(gpt_response, prompt=""): 
    if verbose:
      print (gpt_response)
    try: 
      fields = ["point 1", "point 2", "point 3"]
      gpt_response = extract_first_json_dict(gpt_response)
      for field in fields: 
        if field not in gpt_response: 
          return False
      return True
    except:
      return False 

  def get_fail_safe():
    return None

  prompt_template = f"{prompt_dir}/corr_topics_HyDE_v1.txt" 
  prompt_input = create_prompt_input(name, inquiry, curr_ret_str) 
  prompt = generate_prompt(prompt_input, prompt_template)
  fail_safe = get_fail_safe() 
  output = chat_safe_generate(prompt, "ChatGPT", 5, fail_safe,
                        __chat_func_validate, __chat_func_clean_up, verbose)
  if DEBUG or verbose: 
    print_run_prompts(prompt_template, prompt_input, prompt, output)
  return output, [output, prompt, prompt_input, fail_safe]




def run_gpt_generate_inquiry_response_rpg_new(name, retrieved_obs_str, retrieved_ref_str, inquiry, test_input=None, verbose=False):
  def create_prompt_input(name, retrieved_obs_str, retrieved_ref_str, inquiry, test_input=None):
    prompt_input = [name, name, name, name, 
                    retrieved_obs_str, 
                    name, name, 
                    retrieved_ref_str, 
                    name, name, name,
                    inquiry, 
                    name, name, name, name, name, name, name]
    return prompt_input

  def __chat_func_clean_up(gpt_response, prompt=""): 
    gpt_response = extract_first_json_dict(gpt_response)
    # print ("Thought:", gpt_response["task 0"])
    return gpt_response["task 1"]

  def __chat_func_validate(gpt_response, prompt=""): 
    if verbose:
      print (gpt_response)
    try: 
      fields = ["task 0", "task 1"]
      gpt_response = extract_first_json_dict(gpt_response)
      for field in fields: 
        if field not in gpt_response: 
          return False
      return True
    except:
      return False 

  def get_fail_safe():
    return None

  prompt_template = f"{prompt_dir}/inquiry_response_rpg_v2.txt" 
  prompt_input = create_prompt_input(name, retrieved_obs_str, retrieved_ref_str, inquiry) 
  prompt = generate_prompt(prompt_input, prompt_template)

  # print ("==== DEBUG")
  # print (prompt)
  fail_safe = get_fail_safe() 
  output = chat_safe_generate(prompt, "ChatGPT", 5, fail_safe,
                        __chat_func_validate, __chat_func_clean_up, verbose)
  if DEBUG or verbose: 
    print_run_prompts(prompt_template, prompt_input, prompt, output)
  return output, [output, prompt, prompt_input, fail_safe]



def run_gpt_generate_inquiry_classfier_response(name, retrieved_obs_str, retrieved_ref_str, inquiry, classes, test_input=None, verbose=False):
  def create_prompt_input(name, retrieved_obs_str, retrieved_ref_str, inquiry, classes, test_input=None):
    prompt_input = [inquiry, str(classes), name, name,  
                    retrieved_obs_str, retrieved_ref_str, name, inquiry, str(classes), name]
    return prompt_input

  def __chat_func_clean_up(gpt_response, prompt=""): 
    print ("DEBUGGGG", gpt_response)
    gpt_response = extract_first_json_dict(gpt_response)
    return gpt_response["response"]

  def __chat_func_validate(gpt_response, prompt=""): 
    if verbose:
      print (gpt_response)
    try: 
      fields = ["response"]
      gpt_response = extract_first_json_dict(gpt_response)
      for field in fields: 
        if field not in gpt_response: 
          return False
      return True
    except:
      return False 

  def get_fail_safe():
    return None

  prompt_template = f"{prompt_dir}/inquiry_classification_response_v1.txt" 
  prompt_input = create_prompt_input(name, retrieved_obs_str, retrieved_ref_str, inquiry, classes) 
  prompt = generate_prompt(prompt_input, prompt_template)

  # print ("==== DEBUG")
  # print (prompt)
  fail_safe = get_fail_safe() 
  output = chat_safe_generate(prompt, "ChatGPT", 5, fail_safe,
                        __chat_func_validate, __chat_func_clean_up, verbose)
  if DEBUG or verbose: 
    print_run_prompts(prompt_template, prompt_input, prompt, output)
  return output, [output, prompt, prompt_input, fail_safe]



def run_gpt_generate_inquiry_classfier_response_v2(name, retrieved_obs_str, retrieved_ref_str, inquiry, classes, test_input=None, verbose=False):
  def create_prompt_input(name, retrieved_obs_str, retrieved_ref_str, inquiry, classes, test_input=None):
    prompt_input = [name, name, retrieved_obs_str, retrieved_ref_str,
                    inquiry, str(classes), name, str(classes), name]
    return prompt_input

  def __chat_func_clean_up(gpt_response, prompt=""): 
    print ("DEBUGGGG", gpt_response)
    gpt_response = extract_first_json_dict(gpt_response)
    return gpt_response["response"]

  def __chat_func_validate(gpt_response, prompt=""): 
    if verbose:
      print (gpt_response)
    try: 
      fields = ["response"]
      gpt_response = extract_first_json_dict(gpt_response)
      for field in fields: 
        if field not in gpt_response: 
          return False
      return True
    except:
      return False 

  def get_fail_safe():
    return None

  prompt_template = f"{prompt_dir}/inquiry_classification_response_v2.txt" 
  prompt_input = create_prompt_input(name, retrieved_obs_str, retrieved_ref_str, inquiry, classes) 
  prompt = generate_prompt(prompt_input, prompt_template)

  # print ("==== DEBUG")
  # print (prompt)
  fail_safe = get_fail_safe() 
  output = chat_safe_generate(prompt, "ChatGPT", 5, fail_safe,
                        __chat_func_validate, __chat_func_clean_up, verbose)
  if DEBUG or verbose: 
    print_run_prompts(prompt_template, prompt_input, prompt, output)
  return output, [output, prompt, prompt_input, fail_safe]


def run_gpt_generate_simple_inquire(name, profile, inquiry, test_input=None, verbose=False):
  def create_prompt_input(name, profile, inquiry, test_input=None):
    prompt_input = [name, name,
                    profile, 
                    name, name, name,
                    inquiry, 
                    name, name, name, name, name, name, name]
    return prompt_input

  def __chat_func_clean_up(gpt_response, prompt=""): 
    gpt_response = extract_first_json_dict(gpt_response)
    # print ("Thought:", gpt_response["task 0"])
    return gpt_response["task 1"]

  def __chat_func_validate(gpt_response, prompt=""): 
    if verbose:
      print (gpt_response)
    try: 
      fields = ["task 0", "task 1"]
      gpt_response = extract_first_json_dict(gpt_response)
      for field in fields: 
        if field not in gpt_response: 
          return False
      return True
    except:
      return False 

  def get_fail_safe():
    return None

  prompt_template = f"{prompt_dir}/inquiry_simple_v1.txt" 
  prompt_input = create_prompt_input(name, profile, inquiry) 
  prompt = generate_prompt(prompt_input, prompt_template)

  # print ("==== DEBUG")
  # print (prompt)
  fail_safe = get_fail_safe() 
  output = chat_safe_generate(prompt, "ChatGPT", 5, fail_safe,
                        __chat_func_validate, __chat_func_clean_up, verbose)
  if DEBUG or verbose: 
    print_run_prompts(prompt_template, prompt_input, prompt, output)
  return output, [output, prompt, prompt_input, fail_safe]






def run_gpt_generate_qual_discrete_translation(question, answer, options, test_input=None, verbose=False):
  def create_prompt_input(question, answer, options, test_input=None):
    prompt_input = [question, answer, options]
    return prompt_input

  def __chat_func_clean_up(gpt_response, prompt=""): 
    gpt_response = extract_first_json_dict(gpt_response)["output"]
    return gpt_response

  def __chat_func_validate(gpt_response, prompt=""): 
    if verbose:
      print (gpt_response)
    try: 
      fields = ["output"]
      gpt_response = extract_first_json_dict(gpt_response)
      for field in fields: 
        if field not in gpt_response: 
          return False
      return True
    except:
      return False 

  def get_fail_safe():
    return None

  prompt_template = f"{prompt_dir}/qual_discrete_translation_v1.txt" 
  prompt_input = create_prompt_input(question, answer, options) 
  prompt = generate_prompt(prompt_input, prompt_template)
  # print (prompt)
  fail_safe = get_fail_safe() 
  output = chat_safe_generate(prompt, "ChatGPT", 5, fail_safe,
                        __chat_func_validate, __chat_func_clean_up, verbose)
  if DEBUG or verbose: 
    print_run_prompts(prompt_template, prompt_input, prompt, output)
  return output, [output, prompt, prompt_input, fail_safe]
















def run_gpt_generate_oneshot_ref(name, curr_str, ref_anchor, ref_count, test_input=None, verbose=False):
  def create_prompt_input(name, curr_str, ref_anchor, ref_count, test_input=None):
    prompt_input = [name, curr_str, ref_count, name, ref_anchor, name, name, name]
    return prompt_input

  def __chat_func_clean_up(gpt_response, prompt=""): 
    return extract_first_json_dict(gpt_response)["inferences"]

  def __chat_func_validate(gpt_response, prompt=""): 
    try: 
      print ("hmm....")
      print (extract_first_json_dict(gpt_response)["inferences"])
      return True
    except:
      return False 

  def get_fail_safe():
    return None

  prompt_template = f"{prompt_dir}/oneshot_reflection_v1.txt" 
  prompt_input = create_prompt_input(name, curr_str, ref_anchor, ref_count) 
  prompt = generate_prompt(prompt_input, prompt_template)
  # print (prompt)
  fail_safe = get_fail_safe() 
  output = chat_safe_generate(prompt, "ChatGPT", 5, fail_safe,
                        __chat_func_validate, __chat_func_clean_up, verbose)
  print ("???")
  print (output)
  if DEBUG or verbose: 
    print_run_prompts(prompt_template, prompt_input, prompt, output)
  return output, [output, prompt, prompt_input, fail_safe]




def run_gpt_generate_rerank_ret(name, mem_stream_list, inquiry, test_input=None, verbose=False):
  def create_prompt_input(name, mem_stream_list, inquiry, test_input=None):
    prompt_input = [name, str(mem_stream_list), name, name, inquiry]
    return prompt_input

  def __chat_func_clean_up(gpt_response, prompt=""): 
    return extract_first_json_dict(gpt_response)

  def __chat_func_validate(gpt_response, prompt=""): 
    __chat_func_clean_up(gpt_response)
    try: 
      return True
    except:
      return False 

  def get_fail_safe():
    return None

  prompt_template = f"{prompt_dir}/rerank_retrieved_v1.txt" 
  prompt_input = create_prompt_input(name, mem_stream_list, inquiry) 
  prompt = generate_prompt(prompt_input, prompt_template)
  # print (prompt)
  fail_safe = get_fail_safe() 
  output = chat_safe_generate(prompt, "ChatGPT", 5, fail_safe,
                        __chat_func_validate, __chat_func_clean_up, verbose)
  print ("???")
  print (output)
  if DEBUG or verbose: 
    print_run_prompts(prompt_template, prompt_input, prompt, output)
  return output, [output, prompt, prompt_input, fail_safe]



