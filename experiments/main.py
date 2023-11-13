from utils import *
from global_methods import *
from gpt_structure.gpt_structure import *


def main(): 
  outfile = "output/prelim_test.csv"
  create_folder_if_not_there(outfile)

  personas = ["democrat", 'republican']
  for p in personas: 
    prompt = f"You are a {p}. A hardware store has been selling snow shovels for $15. The morning after a large snowstorm, the store raises the price to $20. Please rate this action as: 1) Completely Fair 2) Acceptable 3) Unfair 4) Very Unfair"
    write_list_to_csv_line([prompt, ChatGPT_simple_request(prompt)], outfile)
    write_list_to_csv_line([prompt, GPT4_simple_request(prompt)], outfile)
    write_list_to_csv_line([prompt, GPT3_instruct_simple_request(prompt)], outfile)
    write_list_to_csv_line([prompt, os_model_simple_request(prompt)], outfile)

    

if __name__ == '__main__':
  main()