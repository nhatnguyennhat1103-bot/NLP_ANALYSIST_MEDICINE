from transformers import pipeline

class NERPredictor:
    def __init__(self, model_path):
        self.nlp = pipeline(
            "token-classification", 
            model=model_path, 
            tokenizer="vinai/phobert-base",
            aggregation_strategy="simple" # Tự động gộp các subwords
        )

    def predict(self, text):
        results = self.nlp(text)
        for entity in results:
            print(f"Thực thể: {entity['word']} | Loại: {id2label[int(entity['entity_group'].split('_')[-1])]}")