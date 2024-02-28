import os
import random
import csv
import tqdm
import argparse
import json
from LM_hf import *
import ipdb

DATA_DIR = './data'
CONDITION_DIR = os.path.join(DATA_DIR, 'conditions')
RESULTS_DIR = os.path.join(DATA_DIR, 'results/representations')
random.seed(0)
    
def ans2belief(belief):
    # Split the sentence into words
    words = belief.split()
    # Extract from the third word and capitalize the first letter of the resulting string
    start = 3 if words[2]=='that' else 2
    statement = ' '.join(words[start:]).capitalize()
    return statement

def ans2action(input_str):
    words = input_str.split()
    if len(words) < 2:
        return input_str.capitalize()
    if words[1].lower() == 'will':
        return ' '.join(words[2:]).capitalize()
    second_word = words[1]
    if second_word.endswith('s') and not second_word.endswith('ss'):
        second_word = second_word[:-1]
    return ' '.join([second_word] + words[2:]).capitalize()

def save_condition(model_name, temperature, method,
                    init_belief, variable, condition, num_probs,
                    max_tokens, verbose, mcq, offset):
    
    with open("./lm_paths.json", "r") as lm_paths:
        paths = json.load(lm_paths)
    llm = LM_nnsight(model_path=paths[model_name])
      
    csv_name = os.path.join(CONDITION_DIR, f'{init_belief}_{variable}_{condition}/stories.csv')
    with open(csv_name, "r") as f:
        reader = csv.reader(f, delimiter=";")
        condition_rows = list(reader)

    if not os.path.exists(os.path.join(RESULTS_DIR, f'{init_belief}_{variable}_{condition}')):
        os.makedirs(os.path.join(RESULTS_DIR, f'{init_belief}_{variable}_{condition}'))
    
    stories = []
    questions = []
    predicted_answers_parsed = []
    graded_answers = []
    answer_keys = []
    thoughts = []
    
    right = 0
    wrong = 0
    anomaly = 0
    tot = num_probs - offset
    # idx = 0

    output_file = os.path.join(RESULTS_DIR, f'{init_belief}_{variable}_{condition}/prompts.csv')
    with open(output_file, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['True', 'False'])
        for idx, row in tqdm.tqdm(enumerate(condition_rows[offset:num_probs])):
            story = row[0]
            true_answer, wrong_answer = row[2], row[3]
            if variable=="belief":
                true_belief = "Story: %s\nBelief: %s" % (story, ans2belief(true_answer))
                false_belief = "Story: %s\nBelief: %s" % (story, ans2belief(wrong_answer))
            elif variable=="action":
                true_belief = "Story: %s\nAction: %s" % (story, ans2action(true_answer))
                false_belief = "Story: %s\nAction: %s" % (story, ans2action(wrong_answer))
            else:
                raise NotImplementedError
            writer.writerow([true_belief, false_belief])

            state_h, state_a = llm.get_all_states(prompt=true_belief)
            # Extract last token
            state_h = state_h[:,-1]
            state_a = state_a[:,-1]
            path_h = os.path.join(RESULTS_DIR, f'{init_belief}_{variable}_{condition}/reps_{method}_{variable}_{condition}_true_{idx}_hidden.npy')
            path_a = os.path.join(RESULTS_DIR, f'{init_belief}_{variable}_{condition}/reps_{method}_{variable}_{condition}_true_{idx}_attention.npy')
            np.save(path_h, state_h)
            np.save(path_a, state_a)

            state_h, state_a = llm.get_all_states(prompt=false_belief)
            # Extract last token
            state_h = state_h[:,-1]
            state_a = state_a[:,-1]
            path_h = os.path.join(RESULTS_DIR, f'{init_belief}_{variable}_{condition}/reps_{method}_{variable}_{condition}_false_{idx}_hidden.npy')
            path_a = os.path.join(RESULTS_DIR, f'{init_belief}_{variable}_{condition}/reps_{method}_{variable}_{condition}_false_{idx}_attention.npy')
            np.save(path_h, state_h)
            np.save(path_a, state_a)      
  
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--variable', type=str, default='belief')
    parser.add_argument('--condition', type=str, default='true_belief')
    parser.add_argument('--model_name', type=str, default='openai/text-davinci-003')
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--num_probs', '-n', type=int, default=1)
    parser.add_argument('--offset', '-o', type=int, default=0)
    parser.add_argument('--max_tokens', type=int, default=100)
    parser.add_argument('--method', type=str, default='0shot')
    parser.add_argument('--init_belief', type=str, default="0_backward")
    parser.add_argument('--verbose', '-v', action='store_true')
    parser.add_argument('--mcq', action='store_true')
    args = parser.parse_args()

    save_condition(args.model_name, args.temperature,
                       args.method, args.init_belief, args.variable,
                       args.condition, args.num_probs, args.max_tokens, args.verbose, args.mcq, args.offset)

if __name__ == '__main__':
    main()