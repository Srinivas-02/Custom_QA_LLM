from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Trainer, TrainingArguments
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import torch

# Load tokenizer and model
model_name = "google/flan-t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# LoRA Configuration
lora_config = LoraConfig(
    r=16,  # Rank of LoRA adaptation
    lora_alpha=32,
    target_modules=["q", "v"],
    lora_dropout=0.1,
    bias="none"
)

# Prepare model for LoRA training
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)

# Load custom data
data_path = "./custom_data.txt"
with open(data_path, "r") as file:
    lines = file.readlines()

# Process custom data
questions, answers = [], []
for line in lines:
    if "---" in line:
        question, answer = line.strip().split("---")
        questions.append(question.strip())
        answers.append(answer.strip())

# Create a dataset
data = {"input_text": questions, "target_text": answers}
dataset = Dataset.from_dict(data)

# Tokenize dataset
def preprocess_function(examples):
    model_inputs = tokenizer(
        examples["input_text"], 
        max_length=512, 
        truncation=True, 
        padding='max_length'
    )
    
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            examples["target_text"], 
            max_length=512, 
            truncation=True, 
            padding='max_length'
        )
    
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Apply tokenization
tokenized_dataset = dataset.map(
    preprocess_function, 
    batched=True, 
    remove_columns=dataset.column_names
)

# Convert to PyTorch dataset
tokenized_dataset.set_format("torch")

# Training Arguments
training_args = TrainingArguments(
    output_dir="./custom_flan_t5_lora",
    per_device_train_batch_size=4,
    num_train_epochs=20,
    learning_rate=1e-5,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    save_strategy="no",
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
)

# Train the model
trainer.train()

# Save LoRA weights
model.save_pretrained("./custom_flan_t5_lora")
tokenizer.save_pretrained("./custom_flan_t5_lora")

print("LoRA Model trained and saved successfully!")