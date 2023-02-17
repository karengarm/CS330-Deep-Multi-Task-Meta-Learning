from cProfile import label
from multiprocessing.sharedctypes import Value
from typing import Dict, List, Optional, Tuple
import torch
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import transformers
import numpy as np
import random

import argparse
from collections import defaultdict
import json
import os
from rouge_score import rouge_scorer
import tqdm

import utils

parser = argparse.ArgumentParser()
parser.add_argument('--task')
parser.add_argument('--model')
parser.add_argument('--dataset')
parser.add_argument('--k')
parser.add_argument('--prompt', default='qa')
parser.add_argument('--debug', action='store_true')
parser.add_argument('--repeats', default=1, type=int)
parser.add_argument('--device', default='cuda')
args = parser.parse_args()


DEVICE = torch.device(args.device)


def get_icl_prompts(
    support_inputs: List[str],
    support_labels: List[str],
    test_input: str,
    prompt_mode: str = 'qa') -> str:
    """
    Take a list of contexts and combine them into k-shot prompts.

    **Note**: Be sure to shuffle the support examples and labels 
      *together* (i.e. so the pairings of support input/label is preserved)
      before constructing the prompt. np.random.permutation may be helpful.

    Args:
      support_inputs: The k inputs used for in-context learning (k may be zero!)
      support_labels: The k labels used for in-context learning (k may be zero!)
      test_input: The input we are evaluating on
      prompt_mode: The task description mode we're using; 'none' means we're only using
        k-shot examples, 'tl;dr' means we're using the tl;dr prompt from the GPT-2 paper,
        etc.

    Returns:
      A string containing the complete input to the model.
    """
    # YOUR CODE HERE
    prompt = ''
    new_prompt = ''
    k = len(support_inputs)

    last_prompt = {"qa": test_input + " In the",
                   'none': test_input,
                   'tldr': test_input + "TL;DR:",
                   'custom': test_input + " summary:"}
    if k != 0:
        #shuffle the order of the support inputs/targets
        zip_input_labels = list(zip(support_inputs, support_labels))
        random.shuffle(zip_input_labels )
        for i, (string_input, string_label) in enumerate(zip_input_labels):
            if prompt_mode == 'qa':
                # Only for bAbI
                # Add “ In the ” after the question and before each answer.
                new_prompt = string_input + " In the " + string_label + "." #string_input.split('.')[-1]
                if i == k - 1:
                    # the input that we want to generate an answer for with a space between
                    # each element (but no space at the end)
                    new_prompt = new_prompt + " " + last_prompt['qa']
            elif prompt_mode == 'none':
                # only for XSum
                # we use the raw k examples without any additional formatting
                new_prompt = string_input + " " + string_label
                if i == k - 1:
                    # the input that we want to generate an answer for with a space between
                    # each element (but no space at the end)
                    new_prompt = new_prompt + " " + last_prompt['none']
            elif prompt_mode == 'tldr':
                # only for XSum
                # Add the text “TL;DR: ” after the article/input and  before the summary/target.
                new_prompt = string_input + "TL;DR: " + string_label
                if i == k - 1:
                    new_prompt = new_prompt + " " + last_prompt['tldr']
            elif prompt_mode == 'custom':
                # only for XSum
                new_prompt = "article:" + string_input + " summary:" + string_label
                if i == k - 1:
                    new_prompt = new_prompt + " article:" + last_prompt['custom']
            # Be sure to include a space
            prompt = prompt + " " + new_prompt
    else:
        prompt = last_prompt[prompt_mode]

    return prompt


def get_performance_metric(predictions: List[str], targets: List[str], metric: str) -> float:
    if metric == 'rouge':
        scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)
        scores = []
        for p, t in zip(predictions, targets):
            score = scorer.score(p, t)['rouge1'].fmeasure
            scores.append(score)
        return sum(scores) / len(scores)
    elif metric == 'exact match':
        if isinstance(targets[0], str):
            return sum([p.strip() == t.strip() for p, t in zip(predictions, targets)]) / len(predictions)
        else:
            def _normalize(prediction):
                if prediction.endswith('Q'):
                    prediction = prediction[:-1]
                elif 'Q:' in prediction:
                    prediction = prediction[:prediction.index('Q:')]
                return prediction.strip('. ').lower()

            normalized = [_normalize(p) for p in predictions]
            def contains(key, candidates):
                for c in candidates:
                    if key in c:
                        return True
                return False

            return sum([contains(n, t) for n, t in zip(normalized, targets)]) / len(normalized)
    else:
        raise NotImplementedError()


def do_sample(model, input_ids, stop_tokens, max_tokens):
    """
    Sample from the model using the given input_ids as a prefix until we either
    hit the stop token or we have sampled max_tokens tokens.

    (Don't use model.generate; implement this yourself in a loop)

    Note: when calling the model here, be sure to wrap the call with
      torch.inferece_mode() to save memory!

    Args:
        model: A transformers.PreTrainedModel that we will sample from.
        input_ids: An integer tensor of shape [1, prefix_len]
        stop_tokens: A list of token ids that indicates that we should stop sampling (e.g., a period)
        max_tokens: Stop sampling if we've sampled this many tokens
    
    Returns:
        The sampled tokens (a python list of ints/zero-dim tensors), not including the input_ids prefix
          OR the stop token (if we hit the stop token before max_tokens)
    """
    # YOUR CODE HERE
    i = 0
    # Stop sampling if we've sampled this many tokens
    while i < max_tokens:
        # Sample from the model using the given input_ids as a prefix
        outputs = model(input_ids, return_dict=True)
        next_token_logits = outputs.logits[:, -1, :]
        # apply a softmax to obtain the probability distribution
        # next_token_probability = torch.softmax(next_token_logits, dim=-1)
        # argmax
        # next_tokens = torch.argmax(next_token_probability, dim=-1)
        # argmax
        next_tokens = torch.argmax(next_token_logits, dim=-1)
        # if next_tokens is in a list of  stop token then we should stop sampling
        #if any(True if k in stop_tokens else False for k in next_tokens):
        if any(True if k in stop_tokens else False for k in next_tokens):
            break
        # Update generated ids, model inputs, and length for next step
        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
        i += 1
    # sampled tokens (a python list of ints/zero-dim tensors), not including the input_ids prefix OR the stop token
    sampled_tokens = input_ids[:, -i:]
    return sampled_tokens[0]


def run_icl(models: List[str], datasets_: List[str], ks: List[int], prompt_modes: List[str], n_val: int = 125):
    results = {}
    for model_name in models:
        print(f'Loading model {model_name}...')
        model, tokenizer = utils.get_model_and_tokenizer(model_name, transformers.AutoModelForCausalLM)
        stop_tokens = utils.stop_tokens(tokenizer)
        model.to(DEVICE)

        for dataset in datasets_:
            print(f'Loading dataset {dataset}...')
            if args.debug:
                n_val = 1
            max_tokens = utils.max_sampled_tokens_for_dataset(dataset)
            train, val = utils.get_dataset(dataset, n_train=max(ks), n_val=n_val)
            for prompt_mode in prompt_modes:
                for k in ks:
                    print(f'Running in-context learning with {model_name} on {dataset} with k={k} and prompt_mode={prompt_mode}')
                    for repeat in range(args.repeats):
                        if repeat > 0:
                            print(f'Beginning repeat #{repeat}')
                        support_idxs = random.choices(range(len(train['x'])), k=k)
                        support_x = [train['x'][idx].replace('\n', ' ') for idx in support_idxs]
                        support_y = [train['simple_y'][idx].replace('\n', ' ') for idx in support_idxs]
                        targets = []
                        predictions = []
                        pbar = tqdm.tqdm(list(range(min(n_val, len(val['x'])))))
                        for row in pbar:
                            test_input = val['x'][row]
                            targets.append(val['y'][row])

                            # Ingredients you'll need:
                            #   get_icl_prompts() [which you implemented]
                            #   do_sample() [which you implemented]
                            #   tokenizer() (for encoding text into tokens) and tokenizer.decode() (for decoding tokens back into text)
                            #   See the documentation for the tokenizer encoder function here:
                            #   https://huggingface.co/docs/transformers/v4.23.1/en/main_classes/tokenizer#transformers.PreTrainedTokenizer.__call__
                            # Note that the tokenizer by default will give you results on the CPU, so you will need to move them to the
                            # proper device.
                            # YOUR CODE HERE
                            prompt = get_icl_prompts(support_x, support_y, test_input, prompt_mode)
                            print(prompt )
                            input_ids = tokenizer(prompt, return_tensors='pt').input_ids
                            input_ids = input_ids.to(DEVICE)
                            outputs = do_sample(model, input_ids, stop_tokens, max_tokens)
                            decoded_prediction = tokenizer.decode(outputs, skip_special_tokens=True)
                            if prompt_mode == 'qa':
                                decoded_prediction = decoded_prediction.split(' ')[-1]
                            # END YOUR CODE

                            predictions.append(decoded_prediction)
                            metric = get_performance_metric(predictions, targets, utils.metric_for_dataset(dataset))
                            pbar.set_description(f'Eval: {metric:.04f}')
                        results['_'.join([model_name, dataset, str(k), prompt_mode])] = metric

                        print('Evaluation results:', results)
                        if not os.path.exists('results/icl'):
                            os.makedirs('results/icl')

                        for k_, v in results.items():
                            with open(f'results/icl/{k_}.json', 'w') as f:
                                json.dump({'metric': v}, f)
                        results = {}


def plot(models, dataset, ks, prompt_modes):
    data = defaultdict(lambda: defaultdict(list))
    symbols = ['solid', 'dashed', 'dotted', 'dashdot']

    x_vals = set()
    for model in models:
        symbol = symbols.pop(0)
        for prompt_mode in prompt_modes:
            for k in ks:
                fn = '_'.join([model, dataset, str(k), prompt_mode])
                id_ = '_'.join([model, dataset, prompt_mode])
                with open(f'results/icl/{fn}.json', 'r') as f:
                    score = json.load(f)['metric']
                    data[id_]['x'].append(k)
                    x_vals.add(k)
                    data[id_]['y'].append(score)
                    data[id_]['linestyle'] = symbol

    for k, v in data.items():
        plt.plot(v['x'], v['y'], label=k, linestyle=v['linestyle'])

    if max(x_vals) > 4:
        plt.xscale('symlog')
    
    ax = plt.gca()
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.xaxis.set_ticks(v['x'])
    plt.legend()
    plt.title(dataset)
    plt.ylabel(utils.metric_for_dataset(dataset))
    plt.xlabel('Number of support examples')
    plt.show()


def run():
    ks = [int(k) for k in args.k.split(',')]
    if args.task == 'icl':
        run_icl(args.model.split(','), args.dataset.split(','), ks, args.prompt.split(','))
    elif args.task == 'plot':
        assert ',' not in args.dataset, "Only one dataset at a time for plotting"
        plot(args.model.split(','), args.dataset, ks, args.prompt.split(','))


if __name__ == '__main__':
    run()
