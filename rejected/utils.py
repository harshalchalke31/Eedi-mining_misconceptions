import json
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

TEMPLATE_INPUT_V3 = '{QUESTION}\nCorrect answer: {CORRECT_ANSWER}\nStudent wrong answer: {STUDENT_WRONG_ANSWER}'

def format_input_v3(row, wrong_choice):
    assert wrong_choice in "ABCD"
    question_text = row.get("QuestionText", "No question text provided")
    subject_name = row.get("SubjectName", "Unknown subject")
    construct_name = row.get("ConstructName", "Unknown construct")
    correct_answer = row.get("CorrectAnswer", "Unknown")
    assert wrong_choice != correct_answer
    correct_answer_text = row.get(f"Answer{correct_answer}Text", "No correct answer text available")
    wrong_answer_text = row.get(f"Answer{wrong_choice}Text", "No wrong answer text available")
    formatted_question = f"Question: {question_text}\n\nSubjectName: {subject_name}\nConstructName: {construct_name}"
    ret = {
        "QUESTION": formatted_question,
        "CORRECT_ANSWER": correct_answer_text,
        "STUDENT_WRONG_ANSWER": wrong_answer_text,
        "MISCONCEPTION_ID": row.get(f'Misconception{wrong_choice}Id'),
    }
    ret["PROMPT"] = TEMPLATE_INPUT_V3.format(**ret)
    return ret

def prepare_items(df, is_submission):
    items = []
    target_ids = []
    for _, row in df.iterrows():
        for choice in ['A', 'B', 'C', 'D']:
            if choice == row["CorrectAnswer"]:
                continue
            if not is_submission and row[f'Misconception{choice}Id'] == -1:
                continue
            item = {'QuestionId_Answer': f"{row['QuestionId']}_{choice}"}
            item['Prompt'] = format_input_v3(row, choice)['PROMPT']
            items.append(item)
            target_ids.append(int(row.get(f'Misconception{choice}Id', -1)))
    return pd.DataFrame(items), target_ids

def get_detailed_instruct(task_description, query):
    return f'<instruct>{task_description}\n<query>{query}'

def get_new_queries(queries, query_max_len, examples_prefix, tokenizer):
    inputs = tokenizer(
        queries,
        max_length=query_max_len - len(tokenizer('<s>', add_special_tokens=False)['input_ids']) -
                    len(tokenizer('\n<response></s>', add_special_tokens=False)['input_ids']),
        return_token_type_ids=False,
        truncation=True,
        return_tensors=None,
        add_special_tokens=False
    )
    prefix_ids = tokenizer(examples_prefix, add_special_tokens=False)['input_ids']
    suffix_ids = tokenizer('\n<response>', add_special_tokens=False)['input_ids']
    new_max_length = (len(prefix_ids) + len(suffix_ids) + query_max_len + 8) // 8 * 8 + 8
    new_queries = tokenizer.batch_decode(inputs['input_ids'])
    for i in range(len(new_queries)):
        new_queries[i] = examples_prefix + new_queries[i] + '\n<response>'
    return new_max_length, new_queries

def save_data(new_queries, documents, output_file="data.json"):
    with open(output_file, 'w') as f:
        data = {'texts': new_queries + documents}
        f.write(json.dumps(data))

def load_qwen_with_lora(base_model_path, lora_path):
    """
    Loads the Qwen model with LoRA integration.
    """
    # Load base model and tokenizer with device_map and torch_dtype to save memory
    model = AutoModelForCausalLM.from_pretrained(base_model_path, torch_dtype=torch.float16, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)

    # Integrate LoRA weights
    model = PeftModel.from_pretrained(model, lora_path)

    return model, tokenizer
