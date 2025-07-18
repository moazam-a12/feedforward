import pickle
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer, Trainer, TrainingArguments
from torch.utils.data import Dataset
import os
import numpy

class FeedbackDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item
    def __len__(self):
        return len(self.labels)

def train_models(project_root, verbose=False):
    trained_models_dir = os.path.join(project_root, 'trained models')
    os.makedirs(trained_models_dir, exist_ok=True)
    
    print("Logistic Regression model is training...")
    with open(os.path.join(project_root, 'utils', 'preprocessed_data_tfidf.pkl'), 'rb') as f:
        data = pickle.load(f)
        X_tfidf = data['X_tfidf']
        y = data['y']
    
    X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)
    lr_model = LogisticRegression(max_iter=1000, random_state=42)
    lr_model.fit(X_train, y_train)
    
    lr_path = os.path.join(trained_models_dir, 'logistic_regression_model.pkl')
    with open(lr_path, 'wb') as f:
        pickle.dump(lr_model, f)
    
    y_pred = lr_model.predict(X_test)
    print("Logistic Regression model is trained.")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred, target_names=['Negative', 'Neutral', 'Positive']))
    cv_scores = cross_val_score(lr_model, X_tfidf, y, cv=5, scoring='accuracy')
    print(f"5-fold CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    print("BERT model is training...")
    with open(os.path.join(project_root, 'utils', 'preprocessed_data_transformer.pt'), 'rb') as f:
        data = torch.load(f, weights_only=False)
        encodings = {'input_ids': data['input_ids'], 'attention_mask': data['attention_mask']}
        labels = data['labels']
        print(f"Encodings input_ids shape: {data['input_ids'].shape}")
        print(f"Labels shape: {labels.shape}")
    
    train_idx, val_idx = train_test_split(range(len(labels)), test_size=0.2, random_state=42)
    train_dataset = FeedbackDataset({k: v[train_idx] for k, v in encodings.items()}, labels[train_idx])
    val_dataset = FeedbackDataset({k: v[val_idx] for k, v in encodings.items()}, labels[val_idx])
    
    model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=3)
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    
    # Ensure model is on MPS device
    device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
    model.to(device)
    
    training_args = TrainingArguments(
        output_dir=os.path.join(trained_models_dir, 'bert_model'),
        num_train_epochs=1,  # Keep as 1 per your original setup
        per_device_train_batch_size=4,  # Reduced for memory efficiency
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,  # Accumulate gradients to simulate batch size of 16
        warmup_steps=100,  # Reduced for faster startup
        weight_decay=0.01,
        logging_strategy='steps',
        logging_steps=50,  # Log more frequently to monitor progress
        eval_strategy='epoch',
        save_strategy='no',  # Keep as is to avoid extra files
        disable_tqdm=False,  # Enable progress bar
        dataloader_pin_memory=False,  # Disable for MPS compatibility
        dataloader_num_workers=4,  # Use multiple workers for data loading
        fp16=False,  # MPS has limited FP16 support; disable unless confirmed working
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=lambda eval_pred: {
            'accuracy': accuracy_score(eval_pred.label_ids, eval_pred.predictions.argmax(-1))
        }
    )
    trainer.train()
    
    bert_path = os.path.join(trained_models_dir, 'bert_model')
    model.save_pretrained(bert_path)
    tokenizer.save_pretrained(bert_path)
    
    print("BERT model is trained.")
    eval_results = trainer.evaluate()
    print(f"BERT evaluation results: {eval_results}")