# !pip install -U transformers
# !pip install transformers datasets torch
import os
os.environ["WANDB_DISABLED"] = "true"

# !pip install -q transformers datasets accelerate torch

import torch
import inspect
from datasets import load_dataset
from transformers import (
    GPT2Tokenizer,
    GPT2LMHeadModel,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)

dataset = load_dataset("imdb", split="train")
dataset = dataset.remove_columns([col for col in dataset.column_names if col != "text"])


model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=64  
    )

tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=["text"]
)

small_dataset = tokenized_dataset.shuffle(seed=42).select(range(2000))

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

model = GPT2LMHeadModel.from_pretrained(model_name)
model.resize_token_embeddings(len(tokenizer))

training_args = TrainingArguments(
    output_dir="./results_en",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    learning_rate=5e-5,
    weight_decay=0.01,
    logging_dir="./logs_en",
    report_to="none",  
    fp16=True,         
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# treinando o modelo 
trainer.train()
trainer.save_model("./results_en/fine_tuned_gpt2_en")

prompt = "Today I felt"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

print("\n=== Generated Texts ===\n")
for i in range(5):
    outputs = model.generate(
        **inputs,
        max_length=80,
        do_sample=True,
        top_k=40,
        top_p=0.92,
        temperature=0.8,
        num_return_sequences=1,
    )
    print(f"Text {i+1}:")
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))
    print("-------------------------------------------------\n")