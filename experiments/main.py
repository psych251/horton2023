from utils import *
from global_methods import *
from gpt_structure.gpt_structure import *


def kahneman_replication(): 
  outfile = "output/kahneman_replication2.csv"
  create_folder_if_not_there(outfile)

  # personas = ["Democratic Party",
  #             "Republican Party",
  #             "Libertarian Party", 
  #             "Green Party", 
  #             "Constitution Party", 
  #             "Independent"]
  personas = ["socialist",
              "leftist",
              "liberal", 
              "moderate", 
              "conservative", 
              "liberterian"]
  surge_prices = ["$16", "$20", "$40", "$100"]
  phrasings = ["changes the price to", 
              "raises the price to"]

  for persona in personas: 
    for surge_price in surge_prices: 
      for phrasing in phrasings: 
        # prompt = f"You are a {persona}. A hardware store has been selling snow shovels for $15. The morning after a large snowstorm, the store {phrasing} {surge_price}. Please rate this action as: 1) Completely Fair 2) Acceptable 3) Unfair 4) Very Unfair"
        prompt = f"You are a {persona}. A hardware store has been selling snow shovels for $15. The morning after a large snowstorm, the store {phrasing} {surge_price}. Please rate this action with one of the following ratings: 'Acceptable', 'Unfair', 'Very Unfair'\n"
        prompt += "output json format: {rating: '<rating>'}"
        write_list_to_csv_line([persona, surge_price, phrasing, prompt, "ChatGPT", ChatGPT_simple_request(prompt)], outfile)
        write_list_to_csv_line([persona, surge_price, phrasing, prompt, "GPT4", GPT4_simple_request(prompt)], outfile)
        write_list_to_csv_line([persona, surge_price, phrasing, prompt, "GPT3", GPT3_instruct_simple_request(prompt)], outfile)
        write_list_to_csv_line([persona, surge_price, phrasing, prompt, "mistral", os_model_simple_request(prompt)], outfile)


def charness_rabin_replication(): 
  outfile = "output/charness_rabin_replication2.csv"
  create_folder_if_not_there(outfile)

  personas = ["",
              "You only care about fairness between players. ",
              "You only care about the total payoff of both players. ",
              "You only care about your own payof. "]
  conditions = ["Option 1: Person B gets $400 and Person A gets $400, Option 2: Person B gets $750 and Person A gets $400",
               "Option 1: Person B gets $0 and Person A gets $800, Option 2: Person B gets $400 and Person A gets $400",
               "Option 1: Person B gets $800 and Person A gets $200, Option 2: Person B gets $0 and Person A gets $0",
               "Option 1: Person B gets $200 and Person A gets $700, Option 2: Person B gets $600 and Person A gets $600",
               "Option 1: Person B gets $300 and Person A gets $600, Option 2: Person B gets $700 and Person A gets $700",
               "Option 1: Person B gets $400 and Person A gets $400, Option 2: Person B gets $750 and Person A gets $375"]

  for persona in personas: 
    for condition in conditions: 
      prompt = f"{persona}Of the following two options, which one would you prefer? {condition}\n"
      prompt += "output json format: {option: '<option>'}"
      write_list_to_csv_line([persona, condition, prompt, "ChatGPT", ChatGPT_simple_request(prompt)], outfile)
      write_list_to_csv_line([persona, condition, prompt, "GPT4", GPT4_simple_request(prompt)], outfile)
      write_list_to_csv_line([persona, condition, prompt, "GPT3", GPT3_instruct_simple_request(prompt)], outfile)
      write_list_to_csv_line([persona, condition, prompt, "mistral", os_model_simple_request(prompt)], outfile)
    

def main():
  kahneman_replication()
  charness_rabin_replication()

if __name__ == '__main__':
  main()