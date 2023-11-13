import json
import random
import openai
import time 

from utils import * 
from langchain.llms import Ollama

openai.api_key = OPENAI_API_KEY


# ============================================================================
# #######################[SECTION 0: HELPER FUNCTIONS] #######################
# ============================================================================

def temp_sleep(seconds=0.1):
  """
  Pause the program's execution for a specified number of seconds.
  ARGS:
    seconds (float, optional): The duration for which the program should sleep,
      in seconds. Default is 0.1 seconds.
  RETURNS:
    None
  Example:
    To pause the program for 2.5 seconds, you can call the function like this:
    >>> temp_sleep(2.5)
  """
  time.sleep(seconds)


def generate_prompt(curr_input, prompt_lib_file): 
  """
  Takes in the current input (e.g. comment that you want to classifiy) and 
  the path to a prompt file. The prompt file contains the raw str prompt that
  will be used, which contains the following substr: !<INPUT>! -- this 
  function replaces this substr with the actual curr_input to produce the 
  final promopt that will be sent to the GPT3 server. 
  ARGS:
    curr_input: the input we want to feed in (IF THERE ARE MORE THAN ONE
                INPUT, THIS CAN BE A LIST.)
    prompt_lib_file: the path to the promopt file. 
  RETURNS: 
    a str prompt that will be sent to OpenAI's GPT server.  
  """
  if type(curr_input) == type("string"): 
    curr_input = [curr_input]
  curr_input = [str(i) for i in curr_input]

  f = open(prompt_lib_file, "r")
  prompt = f.read()
  f.close()
  for count, i in enumerate(curr_input):   
    prompt = prompt.replace(f"!<INPUT {count}>!", i)
  if "<commentblockmarker>###</commentblockmarker>" in prompt: 
    prompt = prompt.split("<commentblockmarker>###</commentblockmarker>")[1]
  return prompt.strip()


# ============================================================================
# ##################### [SECTION 1: SIMPLE GPT CALLERS] ######################
# ============================================================================

def ChatGPT_simple_request(prompt): 
  try: 
    completion = openai.ChatCompletion.create(
      model="gpt-3.5-turbo", 
      messages=[{"role": "user", "content": prompt}])
    output = completion["choices"][0]["message"]["content"]
    return output
  except Exception as e:
    return "GENERATION ERROR"
  

def GPT4_simple_request(prompt): 
  try: 
    completion = openai.ChatCompletion.create(
      model="gpt-4", 
      messages=[{"role": "user", "content": prompt}])
    output = completion["choices"][0]["message"]["content"]
    return output
  except Exception as e:
    return "GENERATION ERROR"


def GPT3_instruct_simple_request(prompt, max_tokens=1000): 
  try: 
    response = openai.Completion.create(
                  model="gpt-3.5-turbo-instruct",
                  prompt=prompt,
                  max_tokens=1000)
    output = response.choices[0].text
    return output
  except Exception as e:
    return "GENERATION ERROR"


def os_model_simple_request(prompt, model="mistral"): 
  try: 
    ollama = Ollama(base_url="http://localhost:11434", model=model)
    response = ollama(prompt)
    return response
  except Exception as e:
    return "GENERATION ERROR"


# ============================================================================
# ####################### [SECTION 2: SAFE GENERATE] #########################
# ============================================================================

def chat_safe_generate(prompt, 
                       gpt_version="GPT4",
                       repeat=3,
                       fail_safe_response="error",
                       func_validate=None,
                       func_clean_up=None,
                       verbose=False): 
  if verbose: 
    print (f"CURRENT PROMPT\n{prompt}")

  for i in range(repeat): 
    try:
      # Right now, we are supporting two end points: ChatGPT and GPT4.
      if gpt_version == "ChatGPT": 
        curr_gpt_response = ChatGPT_simple_request(prompt).strip()
      else: 
        curr_gpt_response = GPT4_simple_request(prompt).strip()

      if curr_gpt_response == "GENERATION ERROR": 
        time.sleep(2**i)
      
      if func_validate(curr_gpt_response, prompt=prompt): 
        return func_clean_up(curr_gpt_response, prompt=prompt)

      print (f"---- REPEAT COUNT: {i}")
      print (f"Current response: ")
      print (curr_gpt_response)
      print (f"---- DEBUG CAES ^: {i}")

    except:
      pass

  print ("FAIL SAFE TRIGGERED") 
  return fail_safe_response


def nonchat_safe_generate(prompt, 
                          max_tokens=1000,
                          gpt_version="gpt-3.5-turbo-instruct",
                          repeat=3,
                          fail_safe_response="error",
                          func_validate=None,
                          func_clean_up=None,
                          verbose=False): 
  if verbose: 
    print (f"CURRENT PROMPT\n{prompt}")

  for i in range(repeat): 
    try:
      # For now, we are assuming gpt-3.5-turbo-instruct in all instances. 
      curr_gpt_response = GPT3_instruct_simple_request(prompt, max_tokens)

      if curr_gpt_response == "GENERATION ERROR": 
        time.sleep(2**i)
      
      if func_validate(curr_gpt_response, prompt=prompt): 
        return func_clean_up(curr_gpt_response, prompt=prompt)

      print (f"---- REPEAT COUNT: {i}")
      print (f"Current response: ")
      print (curr_gpt_response)
      print (f"---- DEBUG CAES ^: {i}")

    except:
      pass

  print ("FAIL SAFE TRIGGERED") 
  return fail_safe_response


def os_model_safe_generate(prompt,
                           model="mistral", 
                           repeat=3,
                           fail_safe_response="error", 
                           func_validate=None,
                           func_clean_up=None,
                           verbose=False): 
  # Current mistral id -> d364aa8d131e
  if verbose: 
    print (f"CURRENT PROMPT\n{prompt}")

  for i in range(repeat): 
    try:
      # For now, we are assuming gpt-3.5-turbo-instruct in all instances. 
      curr_gpt_response = os_model_simple_request(prompt, model)
      
      if func_validate(curr_gpt_response, prompt=prompt): 
        return func_clean_up(curr_gpt_response, prompt=prompt)

      print (f"---- REPEAT COUNT: {i}")
      print (f"Current response: ")
      print (curr_gpt_response)
      print (f"---- DEBUG CAES ^: {i}")

    except:
      pass

  print ("FAIL SAFE TRIGGERED") 
  return fail_safe_response
  

# ============================================================================
# #################### [SECTION 3: OTHER API FUNCTIONS] ######################
# ============================================================================

def get_embedding(text, model="text-embedding-ada-002"):
  text = text.replace("\n", " ")
  if not text: 
    text = "this is blank"
  return openai.Embedding.create(
          input=[text], model=model)['data'][0]['embedding']















