from transformers import AutoTokenizer, AutoModelForCausalLM

model_name_or_path = "./pt_save_pretrained" #path/to/your/model/or/name/on/hub
device = "cuda" # or "cuda" if you have a GPU
model = AutoModelForCausalLM.from_pretrained(model_name_or_path).to(device)
tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m")

def generate(prompt):
    inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)
    outputs = model.generate(inputs,max_new_tokens=150)
    return tokenizer.decode(outputs[0]).replace("### Answer: ","").split('\n')[1].split('.')[0]