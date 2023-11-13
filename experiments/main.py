from utils import *
from gpt_structure.gpt_structure import *


def main(): 
  personas = ["democrat", 'republican']
  for p in personas: 
    prompt = f"You are a {p}. A hardware store has been selling snow shovels for $15. The morning after a large snowstorm, the store raises the price to $20. Please rate this action as: 1) Completely Fair 2) Acceptable 3) Unfair 4) Very Unfair"
    print (ChatGPT_simple_request(prompt))
    print ("--")
    print (GPT4_simple_request(prompt))
    print ("--")
    print (GPT3_instruct_simple_request(prompt))
    print ("--")
    print (os_model_simple_request(prompt))
    print ("--")


if __name__ == '__main__':
  main()