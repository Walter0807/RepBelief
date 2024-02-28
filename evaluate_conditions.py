import os
import random
import csv
import tqdm
import argparse

from evaluate_llm import EvaluateLLM
from evaluate_llm import parse_chat_response
from LM_hf import *
import ipdb
import json

from utils import *

DATA_DIR = 'data'

CONDITION_DIR = os.path.join(DATA_DIR, 'conditions')
RESULTS_DIR = os.path.join(DATA_DIR, 'results/full')
# PROMPT_DIR = '../prompt_instructions'
random.seed(0)
    
def parse_answer(parsed_answer):
    search_for = 'Answer: '
    # Finding the position where 'Answer: ' ends
    occ = parsed_answer.rfind(search_for)
    if occ >= 0:
        return parsed_answer[occ + len(search_for):].strip().rstrip('</s>')
    else:
        return 'Not found.'

def evaluate_condition(model_name, temperature, method,
                    init_belief, variable, condition, num_probs,
                    max_tokens, verbose, mcq, offset, args):

    with open("./lm_paths.json", "r") as lm_paths:
        paths = json.load(lm_paths)
    llm = LM_nnsight(model_path=paths[model_name])
    test_model = EvaluateLLM(llm, method=method)
  
    csv_name = os.path.join(CONDITION_DIR, f'{init_belief}_{variable}_{condition}/stories.csv')
    with open(csv_name, "r") as f:
        reader = csv.reader(f, delimiter=";")
        condition_rows = list(reader)

    stories = []
    questions = []
    predicted_answers_parsed = []
    flags_correct = []
    flags_invalid = []
    answer_keys = []
    thoughts = []
    
    right = 0
    wrong = 0
    anomaly = 0
    tot = num_probs - offset
    idx = 0

    # Load intervention dict
    if 'interv' in test_model.method:
        test_model.load_interv(args)

    for row in tqdm.tqdm(condition_rows[offset:num_probs]):
        idx += 1
        story = row[0]
        question_orig = row[1]
        question = row[1]
        true_answer, wrong_answer = row[2], row[3]
        answers = [true_answer, wrong_answer]
        # ipdb.set_trace()
        random.shuffle(answers)
        if mcq:
            question = f"{question}\nChoose one of the following:\na) {answers[0]}\nb) {answers[1]}"
        predicted_answer, thought = test_model.predict_answer(story, question, args)
        if answers[0] == true_answer:
            answer_key = 'a)'
            negative_answer_key = 'b)'
        else:
            answer_key = 'b)'
            negative_answer_key = 'a)'
        if mcq:         
            predicted_answer_parsed = parse_answer(predicted_answer)
        correct = (predicted_answer_parsed[:2].lower()==answer_key)
        incorrect = (predicted_answer_parsed[:2].lower()==negative_answer_key)
        not_found = (not correct) and (not incorrect)

        if not_found:
            # Double check the answer.
            choose_right = (answer_key in predicted_answer_parsed) or (true_answer.lower().rstrip('.') in predicted_answer_parsed.lower())
            choose_wrong = (negative_answer_key in predicted_answer_parsed) or (wrong_answer.lower().rstrip('.') in predicted_answer_parsed.lower())
            if choose_right and (not choose_wrong):
                not_found = 0
                correct = 1
            if choose_wrong and (not choose_right):
                not_found = 0
                incorrect = 1

        correct, not_found = int(correct), int(not_found)
        
        right += correct
        wrong += incorrect
        anomaly += not_found

        if verbose:
            print_colored(f"THOUGHT: {thought}", "blue")
            print_colored(f"PREDICT: {predicted_answer_parsed}", "green")
            print(f"RIGHT: {answer_key} {true_answer}")
            print(f"WRONG: {negative_answer_key} {wrong_answer}")
            print_colored(f"GRADE: Right={correct}, Invalid={not_found}", "yellow")
            
        print_colored(f"Right: Wrong: Anomaly = ({right}: {wrong}: {anomaly}) / {idx}. Acc: {right / idx:.2f}", "red")

        
        stories.append(story)
        questions.append(question)
        predicted_answers_parsed.append(predicted_answer_parsed)
        flags_correct.append(correct)
        flags_invalid.append(not_found)
        answer_keys.append(answer_key)
        thoughts.append(thought)
    # save results
    model_name = model_name.replace('/', '_')
    prediction = os.path.join(RESULTS_DIR, f'{init_belief}_{variable}_{condition}/Dir_{args.direction}_{args.dynamic}_{args.dvariable}/prediction_{model_name}_{temperature}_{method}_{variable}_{condition}_{offset}_{num_probs}.csv')

    if not os.path.exists(os.path.join(RESULTS_DIR, f'{init_belief}_{variable}_{condition}/Dir_{args.direction}_{args.dynamic}_{args.dvariable}/')):
        os.makedirs(os.path.join(RESULTS_DIR, f'{init_belief}_{variable}_{condition}/Dir_{args.direction}_{args.dynamic}_{args.dvariable}/'))
    
    combined_results = zip(stories, questions, answer_keys, predicted_answers_parsed, flags_correct, flags_invalid, thoughts)
    
    with open(prediction, "w") as f:
        writer = csv.writer(f, delimiter=";")
        # Write a header row
        writer.writerow(["Story", "Question", "Correct Answer", "Predicted Answer", "Correct", "Invalid", "Thought"])
        for row in combined_results:
            writer.writerow(row)

    accuracy = right / len(flags_correct)

    # Print results
    print("\n------------------------")
    print("         RESULTS        ")
    print("------------------------")
    print(f"MODEL: {model_name}, Temperature: {temperature}, Method: {method}")
    print(f"CONDITION: {init_belief} {variable}, {condition}")
    print(f"ACCURACY: {accuracy:.2%}")
    print("------------------------\n")
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--variable', type=str, default='belief')
    parser.add_argument('--dvariable', type=str, default='None')
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

    parser.add_argument('--belief', type=str, default='protagonist')
    parser.add_argument('--dynamic', type=str, default='0_forward')
    parser.add_argument('--K', type=int, default=32)
    parser.add_argument('--alpha', type=int, default=20)
    parser.add_argument('--direction', type=str, default="CoM")
    parser.add_argument('--frac', type=float, default=0.)
    args = parser.parse_args()
    print("\nParameters:")
    for attr, value in sorted(args.__dict__.items()):
        print("\t{}={}".format(attr.upper(), value))

    if args.dvariable=='None':
            args.dvariable = args.variable
    if 'interv' in args.method:
        args.method += "_%s_K%d_a%d_%s" % (args.belief, args.K, args.alpha, args.direction)
    
    evaluate_condition(args.model_name, args.temperature,
                       args.method, args.init_belief, args.variable,
                       args.condition, args.num_probs, args.max_tokens, args.verbose, args.mcq, args.offset, args)

if __name__ == '__main__':
    main()