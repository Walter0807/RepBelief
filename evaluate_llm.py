import torch
import numpy as np
import ipdb
from utils import *
from probe import load_data


def parse_chat_response(response):
    answer_idx = response.find('Answer:')
    return response[answer_idx+8:].strip().rstrip('</s>')

class EvaluateLLM():

    def __init__(self, llm, method='0shot'):
        self.llm = llm
        self.instruction = None
        self.method = method

        if method == '0shot' or '0shot-interv' in method:
            # predict answer
            self.stop_tokens = ["Story:", "Question:"]
            with(open('instructions.txt', 'r')) as f:
                self.instruction = f.read()
            self.prompt = """{instruction}

Story: {story}
Question: {question}
Answer:"""
        else:
            raise ValueError(f"method {method} not supported")
    
    def load_interv(self, args):
        probe_dir = "./data/results/probe"
        val_acc_all = np.load("%s/%s_%s_%s_val_acc.npy" % (probe_dir, args.dynamic, args.dvariable, args.belief))
        coefs_all = np.load("%s/%s_%s_%s_coef.npy" % (probe_dir, args.dynamic, args.dvariable, args.belief))
        
        all_activations, _ = load_data(dynamic=args.init_belief, belief=args.belief, variable=args.variable) # load test domain activations
    
        coefs_oracle = np.load("%s/%s_%s_oracle_coef.npy" % (probe_dir, args.dynamic, args.dvariable))
        coefs_protagonist = np.load("%s/%s_%s_protagonist_coef.npy" % (probe_dir, args.dynamic, args.dvariable))

        coefs_multinomial = np.load("%s/%s_%s_multinomial_coef.npy" % (probe_dir, args.dynamic, args.dvariable))
        val_acc_multinomial = np.load("%s/%s_%s_multinomial_val_acc.npy" % (probe_dir, args.dynamic, args.dvariable))
        # 0 (o0p0), 1 (o0p1), 2 (o1p0), 3 (o1p1)
    
        # normalize the directions.
        coefs_all = normalize_vectors(coefs_all)
        coefs_multinomial = normalize_vectors(coefs_multinomial)
    
        if 'multi' in args.direction:
            top_heads = find_largest_k_items(val_acc_multinomial, args.K)
        else:
            top_heads = find_largest_k_items(val_acc_all, args.K)
    
        # Single Logistic Regression Directions
        if args.direction == "Coef":
            directions = coefs_all
        # Multinomial Logistic Regression Directions
        elif args.direction == "multi_o0p1":
            directions = coefs_multinomial[:,:,1,:]
        elif args.direction == "random":
            shape = coefs_multinomial.shape
            np.random.seed(args.seed)
            random_array = np.random.randn(shape[0], shape[1], shape[3])
            directions = normalize_vectors(random_array)
        else:
            raise NotImplementedError
        self.interventions_dict = get_interventions_dict(all_activations, top_heads, directions=directions)
        
    def predict_answer(self, story, question, args):
        if 'interv' in self.method:
            prompt = self.prompt.format(instruction=self.instruction, story=story, question=question)
            response = self.llm.intervention(prompt, self.interventions_dict, alpha=args.alpha, max_new_tokens=args.max_tokens)
        else:
            prompt = self.prompt.format(instruction=self.instruction, story=story, question=question)
            response = self.llm(prompt=prompt)
        return response, -1