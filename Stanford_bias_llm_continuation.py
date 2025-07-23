


import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from scipy.stats import ttest_ind

model_names = [
    "gpt2",                       
    "distilgpt2",                
    "gpt2-medium",               
    "gpt2-large",                
    "EleutherAI/gpt-neo-125M",   
    "EleutherAI/gpt-neo-1.3B",   
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizers = {name: AutoTokenizer.from_pretrained(name) for name in model_names}
models = {}
for name in model_names:
    model = AutoModelForCausalLM.from_pretrained(name)
    model.to(device)
    model.eval()
    models[name] = model

df = pd.read_csv("ayman_stanford2.csv")
df['Slant'] = pd.to_numeric(df['Slant'], errors='coerce')  

def compute_log_likelihood(model_name, text):
    tokenizer = tokenizers[model_name]
    model = models[model_name]
    inputs = tokenizer(text, return_tensors="pt").to(device)
    input_ids = inputs.input_ids
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        neg_log_likelihood = outputs.loss.item() * input_ids.shape[1]
    return -neg_log_likelihood, input_ids.shape[1]

results = []

for model_name in model_names:
    tokenizer = tokenizers[model_name]
    context_window = tokenizer.model_max_length
    print(f"Processing with model: {model_name} | Context Window: {context_window} tokens")
    
    for _, row in df.iterrows():
        full_text = row["Text2"]
        slant = row["Slant"]
        question = row["Question"]
        
        if pd.isna(slant) or not full_text or not question:
            continue
            
        expected_preamble = f"{question} is an important issue that is best addressed by".strip().lower()
        
        # Extract preamble portion from full_text by locating "is an important issue..."
        keyword = "is an important issue that is best addressed by"
        idx = full_text.lower().find(keyword)
        
        if idx == -1:
            print(f"Skipping malformed row for model {model_name}: Could not locate preamble")
            continue
        
        preamble = full_text[:idx + len(keyword)].strip()
        
        try:
            ll_full, tokens_full = compute_log_likelihood(model_name, full_text)
            ll_preamble, tokens_preamble = compute_log_likelihood(model_name, preamble)
            
            continuation_ll = ll_full - ll_preamble
            continuation_tokens = tokens_full - tokens_preamble
            per_token_ll = continuation_ll / continuation_tokens if continuation_tokens > 0 else None
            
            results.append({
                "Question": question,
                "Numerical_Slant": slant,
                "Model": model_name,
                "Full_Log_Likelihood": ll_full,
                "Preamble_Log_Likelihood": ll_preamble,
                "Continuation_Log_Likelihood": continuation_ll,
                "Continuation_Token_Count": continuation_tokens,
                "Per_Token_Continuation_LL": per_token_ll
            })
        except Exception as e:
            print(f"Error processing row with {model_name}: {str(e)}")

results_df = pd.DataFrame(results)
results_df.to_csv("political_text_continuation_likelihoods.csv", index=False)
print("Saved continuation likelihoods to political_text_continuation_likelihoods.csv")

