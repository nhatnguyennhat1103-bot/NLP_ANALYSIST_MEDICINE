from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer

def train_model():
    model = AutoModelForTokenClassification.from_pretrained(
        "vinai/phobert-base", 
        num_labels=len(label_list)
    )

    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        num_train_epochs=10,
        weight_decay=0.01,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset, # Khởi tạo từ class ở Bước 2
        eval_dataset=dev_dataset,
    )
    
    trainer.train()
    model.save_pretrained("src/saved_model")