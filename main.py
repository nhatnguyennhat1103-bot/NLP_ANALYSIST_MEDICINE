import os
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer
from underthesea import word_tokenize

# Import các class từ thư mục src của bạn
from src.dataset import MedicalNERDataset
# Giả sử bạn để logic dự đoán trong src/inference.py
# from src.inference import NERPredictor 

def load_labels(label_path):
    with open(label_path, 'r', encoding='utf-8') as f:
        labels = [line.strip() for line in f if line.strip()]
    label2id = {label: i for i, label in enumerate(labels)}
    id2label = {i: label for i, label in enumerate(labels)}
    return labels, label2id, id2label

def main():
    # --- 1. Cấu hình thông số ---
    MODEL_NAME = "vinai/phobert-base"
    DATA_DIR = "data/processed"
    LABEL_FILE = "data/processed/labels.txt"
    SAVE_DIR = "src/saved_model"
    MAX_LEN = 128
    BATCH_SIZE = 16
    EPOCHS = 10
    
    # Kiểm tra thiết bị (Ưu tiên GPU nếu có)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Đang chạy trên thiết bị: {device}")

    # --- 2. Chuẩn bị Label và Tokenizer ---
    label_list, label2id, id2label = load_labels(LABEL_FILE)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # --- 3. Tải Dữ liệu ---
    print("📦 Đang tải dữ liệu...")
    train_dataset = MedicalNERDataset(os.path.join(DATA_DIR, "train.txt"), tokenizer, label2id, MAX_LEN)
    dev_dataset = MedicalNERDataset(os.path.join(DATA_DIR, "dev.txt"), tokenizer, label2id, MAX_LEN)

    # --- 4. Khởi tạo Mô hình ---
    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME, 
        num_labels=len(label_list),
        id2label=id2label,
        label2id=label2id
    ).to(device)

    # --- 5. Huấn luyện (Chỉ chạy khi cần) ---
    SHOULD_TRAIN = True # Đổi thành False nếu bạn đã có model và muốn chạy Predict ngay
    
    if SHOULD_TRAIN:
        print("🛠️ Bắt đầu quá trình huấn luyện...")
        training_args = TrainingArguments(
            output_dir="./results",
            evaluation_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=BATCH_SIZE,
            per_device_eval_batch_size=BATCH_SIZE,
            num_train_epochs=EPOCHS,
            weight_decay=0.01,
            save_total_limit=2,
            logging_dir='./logs',
            push_to_hub=False,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=dev_dataset,
        )

        trainer.train()
        model.save_pretrained(SAVE_DIR)
        tokenizer.save_pretrained(SAVE_DIR)
        print(f"✅ Đã lưu mô hình tại: {SAVE_DIR}")

    # --- 6. Chạy thử nghiệm dự đoán (Inference) ---
    print("\n🔍 Chạy thử nghiệm dự đoán:")
    test_sentence = "Bệnh nhân có triệu chứng đau dạ dày và đã dùng thuốc Omeprazole để điều trị."
    
    # Bước quan trọng: Tách từ tiếng Việt trước khi đưa vào mô hình
    segmented_sentence = word_tokenize(test_sentence, format="text")
    
    # Dùng pipeline của Hugging Face để dự đoán nhanh
    from transformers import pipeline
    nlp_ner = pipeline("token-classification", model=model, tokenizer=tokenizer, aggregation_strategy="simple")
    
    ner_results = nlp_ner(segmented_sentence)

    print(f"Câu gốc: {test_sentence}")
    print("-" * 30)
    for ent in ner_results:
        # Làm sạch dấu gạch dưới từ word_tokenize để hiển thị đẹp hơn
        word = ent['word'].replace('_', ' ')
        label = ent['entity_group']
        print(f"Thực thể: {word:20} | Nhãn: {label}")

if __name__ == "__main__":
    main()