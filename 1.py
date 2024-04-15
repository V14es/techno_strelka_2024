import transformers
from datasets import load_dataset, load_metric
import nltk
import string
from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer, AutoTokenizer

model_checkpoint = "t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
batch_size = 8

model_name = "t5-base-medium-title-generation"
model_dir = f"Models/{model_name}"

args = Seq2SeqTrainingArguments(model_dir,
    evaluation_strategy="steps",
    eval_steps=100,
    logging_strategy="steps",
    logging_steps=100,
    save_strategy="steps",
    save_steps=200,
    learning_rate=4e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=1,
    predict_with_generate=True,
    #fp16=True,
    load_best_model_at_end=True,
    metric_for_best_model="rouge1",
    report_to="tensorboard"
)
'''def clean_text(text):
  sentences = nltk.sent_tokenize(text.strip())
  sentences_cleaned = [s for sent in sentences for s in sent.split("\n")]
  sentences_cleaned_no_titles = [sent for sent in sentences_cleaned
                                 if len(sent) > 0 and
                                 sent[-1] in string.punctuation]
  text_cleaned = "\n".join(sentences_cleaned_no_titles)
  return text_cleaned

def preprocess_data(examples):
  texts_cleaned = [clean_text(text) for text in examples["text"]]
  inputs = [prefix + text for text in texts_cleaned]
  model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)

  # Setup the tokenizer for targets
  with tokenizer.as_target_tokenizer():
    labels = tokenizer(examples["title"], max_length=max_target_length,
                       truncation=True)

  model_inputs["labels"] = labels["input_ids"]
  return model_inputs
'''

medium_datasets = load_dataset("IlyaGusev/ru_turbo_saiga", trust_remote_code=True)

print(medium_datasets)

datasets_train_test = medium_datasets["train"].train_test_split(test_size=2000)
datasets_train_validation = datasets_train_test["train"].train_test_split(test_size=2000)

medium_datasets["train"] = datasets_train_validation["train"] #основная выборка
medium_datasets["validation"] = datasets_train_validation["test"] #проверка
medium_datasets["test"] = datasets_train_test["test"] #для тестирования

#tokenized_datasets = medium_datasets.map(batched=True)

print(medium_datasets)