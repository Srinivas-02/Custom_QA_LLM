from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Trainer, TrainingArguments
from datasets import Dataset
import torch

# Load tokenizer and model
model_name = "google/flan-t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Load custom data
data_path = "./custom_data.txt"  # Path to your text file
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
    
    # Tokenize targets with the same padding
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            examples["target_text"], 
            max_length=512, 
            truncation=True, 
            padding='max_length'
        )
    
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Apply tokenization with batched=True
tokenized_dataset = dataset.map(
    preprocess_function, 
    batched=True, 
    remove_columns=dataset.column_names
)

# Convert to PyTorch dataset
tokenized_dataset.set_format("torch")

# Set up training arguments
training_args = TrainingArguments(
    output_dir="./custom_flan_t5",
    per_device_train_batch_size= 16,
    num_train_epochs=20,
    save_steps=99999,
    save_strategy="no",
    save_total_limit=0,
    evaluation_strategy="no",
    logging_dir="./logs",
    logging_steps=10,
)

# Set up Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
)

# Train the model
trainer.train()

# Save the model and tokenizer
trainer.save_model("./custom_flan_t5")
tokenizer.save_pretrained("./custom_flan_t5")

print("Model trained and saved successfully!")