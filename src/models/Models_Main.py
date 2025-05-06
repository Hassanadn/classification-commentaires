from src.data.load_data import DataLoader
from BertTextClassifier import BertTextClassifier

if __name__ == "__main__":
    config_path = './config/config.yaml'
    data_loader = DataLoader(config_path)
    trainer = BertTextClassifier(config_path)

    for i, chunk_df in enumerate(data_loader.chunk_generator()):
        print(f"=== Entraînement sur le chunk {i+1} ===")
        trainer.train_on_chunk(chunk_df, chunk_num=i+1)

    # Sauvegarde finale du modèle et du tokenizer
    final_model_path = "/content/models/final_bert"
    trainer.save_model(trainer.model, final_model_path)
    trainer.feature_engineer.tokenizer.save_pretrained(final_model_path)
