from collections import Counter

import evaluate
import pandas as pd
from tabulate import tabulate


metric_dict_star = {
    "Interaction": Counter(),
    "Sequence": Counter(),
    "Prediction": Counter(),
    "Feasibility": Counter(),
    "Average": Counter()
}

metric_dict_agqa = {
    "Reasoning":
    {   "obj-rel":
            {"binary": Counter(), "open": Counter(), "all": Counter()},
        "rel-act":
            {"binary": Counter()},
        "obj-act":
            {"binary": Counter()},
        "superlative":
            {"binary": Counter(), "open": Counter(), "all": Counter()},
        "sequencing":
            {"binary": Counter(), "open": Counter(), "all": Counter()},
        "exists":
            {"binary": Counter()},
        "duration-comparison":
            {"binary": Counter(), "open": Counter(), "all": Counter()},
        "action-recognition":
            {"open": Counter()}
    },
    "Semantic":
    {
        "object":
            {"binary": Counter(), "open": Counter(), "all": Counter()},
        "relation":
            {"binary": Counter()},
        "action":
            {"binary": Counter(), "open": Counter(), "all": Counter()},
    },
    "Structure":
    {
        "query":
            {"open": Counter()},
        "compare":
            {"binary": Counter()},
        "choose":
            {"binary": Counter()},
        "logic":
            {"binary": Counter()},
        "verify":
            {"binary": Counter()}
    },
    "Overall":
    {
        "binary": Counter(),
        "open": Counter(),
        "all": Counter()
    }
}

def process_nested_dict(nested_dict, parent_keys=None):
    """
    Recursively processes the nested dictionary, printing the key path and ratio if value is a Counter.
    """
    if parent_keys is None:
        parent_keys = []  # Initialize the parent_keys list

    table_data = []
    for key, value in nested_dict.items():
        current_keys = parent_keys + [key]

        if isinstance(value, Counter):  # Check if the value is a Counter
            ratio = calculate_ratio(value)
            row = current_keys + [f"{round(ratio, 2)} ({value})"]
            table_data.append(row)
        elif isinstance(value, dict):  # If the value is another dictionary, recurse
            table_data.extend(process_nested_dict(value, current_keys))  # Recurse with updated key path
        else:
            raise NotImplementedError
    return table_data

def calculate_ratio(counter):
    # Get the number of 0's and 1's
    num_zeros = counter.get(0, 0)
    num_ones = counter.get(1, 0)

    # Avoid division by zero and calculate the ratio
    if num_zeros != 0:
        return num_ones / num_zeros
    else:
        return -1  # Returning -1 if there are no zeros

def get_accuracy_star(args):
    assert args.do_template == False, "not Implemented"
    path = f"eval/{args.dataset_name}/{args.exp_name}.json"
    df = pd.read_json(path, lines=True)
    for pred, answer, q_t in zip(df["pred"], df["answer"], df["question_type"]):
        metric_dict_star[q_t].update([0]) #all
        metric_dict_star["Average"].update([0])
        if answer in pred:
            metric_dict_star[q_t].update([1]) #tp
            metric_dict_star["Average"].update([1])

    table_data = process_nested_dict(metric_dict_star)
    print(tabulate(table_data, tablefmt="grid", stralign="left"))

def get_accuracy_agqa(args):
    assert args.do_template == False, "not Implemented"
    path = f"eval/{args.dataset_name}/{args.exp_name}.json"
    df = pd.read_json(path, lines=True)
    for pred, answer, q_t in zip(df["pred"], df["answer"], df["question_type"]):
        verdict = answer in pred

        metric_dict_agqa["Overall"]["all"].update([0])
        if verdict:
            metric_dict_agqa["Overall"]["all"].update([1])

        binary_flag = q_t["bo_type"]
        metric_dict_agqa["Overall"][binary_flag].update([0])
        if verdict:
            metric_dict_agqa["Overall"][binary_flag].update([1])

        reasoning_cat = q_t["reasoning"]
        for r_key in reasoning_cat:
            if "all" in metric_dict_agqa["Reasoning"][r_key]:
                metric_dict_agqa["Reasoning"][r_key]["all"].update([0])
                if verdict:
                    metric_dict_agqa["Reasoning"][r_key]["all"].update([1])
            metric_dict_agqa["Reasoning"][r_key][binary_flag].update([0])
            if verdict:
                metric_dict_agqa["Reasoning"][r_key][binary_flag].update([1])

        semantic_cat = q_t["semantic"]
        if "all" in metric_dict_agqa["Semantic"][semantic_cat]:
            metric_dict_agqa["Semantic"][semantic_cat]["all"].update([0])
            if verdict:
                metric_dict_agqa["Semantic"][semantic_cat]["all"].update([1])
        metric_dict_agqa["Semantic"][semantic_cat][binary_flag].update([0])
        if verdict:
            metric_dict_agqa["Semantic"][semantic_cat][binary_flag].update([1])

        structural_cat = q_t["structural"]
        if "all" in metric_dict_agqa["Structure"][structural_cat]:
            metric_dict_agqa["Structure"][structural_cat]["all"].update([0])
            if verdict:
                metric_dict_agqa["Structure"][structural_cat]["all"].update([1])
        metric_dict_agqa["Structure"][structural_cat][binary_flag].update([0])
        if verdict:
            metric_dict_agqa["Structure"][structural_cat][binary_flag].update([1])

    table_data = process_nested_dict(metric_dict_agqa)
    print(tabulate(table_data, tablefmt="grid", stralign="left"))

def get_accuracy_full(predictions, references):
    assert len(predictions) == len(references)
    
    bleu_metric = evaluate.load("bleu")
    rouge_metric = evaluate.load("rouge")
    meteor_metric = evaluate.load("meteor")
    bertscore_metric = evaluate.load("bertscore")

    accuracy = 0
    for pred, answer in zip(predictions, references):
        if answer in pred:
            accuracy += 1
    accuracy /= len(predictions)

    bleu_results = bleu_metric.compute(predictions=predictions, references=references)
    rouge_results = rouge_metric.compute(predictions=predictions, references=references)
    meteor_results = meteor_metric.compute(predictions=predictions, references=references)
    bertscore_results = bertscore_metric.compute(predictions=predictions, references=references, lang="en")

    return {
            "accuracy": round(accuracy, 2),
            "bleu": round(bleu_results['bleu'], 2),
            "rougeL": round(rouge_results['rougeL'], 2),
            "meteor": round(meteor_results['meteor'], 2),
            "bertscore": round(sum(bertscore_results['f1']) / len(bertscore_results['f1']), 2)
           }
    

eval_funcs = {
    "star": get_accuracy_star,
    "agqa": get_accuracy_agqa,
}
