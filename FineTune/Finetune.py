import os
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration, TrainingArguments, Trainer
from datasets import load_dataset, load_metric
from PIL import Image

# Load the processor and model
model_name = "Salesforce/blip-image-captioning-large"
processor = BlipProcessor.from_pretrained(model_name)
model = BlipForConditionalGeneration.from_pretrained(model_name)

# Load dataset
dataset = load_dataset("jpawan33/kag100-image-captioning-dataset")

# Preprocess the dataset
def preprocess(examples):
    images = [Image.open(image).convert("RGB") for image in examples["image"]]
    inputs = processor(images=images, text=examples["caption"], padding="max_length", truncation=True, return_tensors="pt")
    inputs["labels"] = inputs["input_ids"]
    return inputs

dataset = dataset.map(preprocess, batched=True, remove_columns=["image", "caption"])
dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# Split dataset
train_dataset = dataset["train"]
val_dataset = dataset["validation"]

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01,
    save_total_limit=3,
    save_steps=10,
    logging_dir="./logs",
)

# Define metrics
def compute_metrics(eval_preds):
    preds, labels = eval_preds
    decoded_preds = processor.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = processor.batch_decode(labels, skip_special_tokens=True)
    metric = load_metric("bleu")
    decoded_preds = [pred.split() for pred in decoded_preds]
    decoded_labels = [[label.split()] for label in decoded_labels]
    return metric.compute(predictions=decoded_preds, references=decoded_labels)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=processor,
    compute_metrics=compute_metrics
)

# Fine-tuning the model
trainer.train()

# Save the model
model.save_pretrained("fine-tuned-blip")
processor.save_pretrained("fine-tuned-blip")
