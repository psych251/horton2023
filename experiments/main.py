from utils import *
from global_methods import *
from gpt_structure.gpt_structure import *


def kahneman_replication(): 
  outfile = "output/kahneman_replication.csv"
  create_folder_if_not_there(outfile)

  personas = ["Democratic Party",
              "Republican Party",
              "Libertarian Party", 
              "Green Party", 
              "Constitution Party", 
              "Independent"]

  surge_prices = ["$16", "$20", "$40", "$100"]

  phrasings = ["changes the price to", 
              "raises the price to"]

  for persona in personas: 
    for surge_price in surge_prices: 
      for phrasing in phrasings: 
        prompt = f"You are a {p}. A hardware store has been selling snow shovels for $15. The morning after a large snowstorm, the store {phrasing} {surge_price}. Please rate this action as: 1) Completely Fair 2) Acceptable 3) Unfair 4) Very Unfair"
        write_list_to_csv_line([persona, surge_price, phrasing, prompt, ChatGPT_simple_request(prompt)], outfile)
        write_list_to_csv_line([persona, surge_price, phrasing, prompt, GPT4_simple_request(prompt)], outfile)
        write_list_to_csv_line([persona, surge_price, phrasing, prompt, GPT3_instruct_simple_request(prompt)], outfile)
        write_list_to_csv_line([persona, surge_price, phrasing, prompt, os_model_simple_request(prompt)], outfile)


def charness_rabin_replication(): 
  outfile = "output/charness_rabin_replication.csv"
  create_folder_if_not_there(outfile)

  condition = ["Person B gets $600 and Person A gets $300", "Person B gets $500 and Person A gets $700"]
  prompt = f"{condition[0]}, {condition[1]}"
  return 
    
def main():
  kahneman_replication()
  charness_rabin_replication()

if __name__ == '__main__':
  main()