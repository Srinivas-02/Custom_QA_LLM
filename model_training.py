import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Trainer, TrainingArguments
from datasets import Dataset

# Check if CUDA is available and set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load tokenizer and model
model_name = "google/flan-t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)

# Load custom data
data_path = "custom_data.txt"
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

# Define the task-specific prefix
prefix = "answer the question: "

# Tokenize dataset
def preprocess_function(examples):
    inputs = [prefix + question for question in examples["input_text"]]
    model_inputs = tokenizer(
        inputs, max_length=512, truncation=True, padding="max_length"
    )
    labels = tokenizer(
        examples["target_text"], max_length=512, truncation=True, padding="max_length"
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Apply tokenization
tokenized_dataset = dataset.map(
    preprocess_function, batched=True, remove_columns=dataset.column_names
)

# Training Arguments
training_args = TrainingArguments(
    output_dir="./custom_flan_t5",
    per_device_train_batch_size=4,
    num_train_epochs=3,
    learning_rate=5e-5,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    save_strategy="epoch",
    evaluation_strategy="epoch",
    load_best_model_at_end=True,
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    eval_dataset=tokenized_dataset,
)

# Train the model
trainer.train()

# Save the fine-tuned model and tokenizer
model.save_pretrained("./custom_flan_t5")
tokenizer.save_pretrained("./custom_flan_t5")

print("Model fine-tuned and saved successfully!")
