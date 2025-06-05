import pandas as pd
import tensorflow as tf
from transformers import TFT5ForConditionalGeneration, T5Tokenizer
from sklearn.model_selection import train_test_split

# Sample training data (replace with your dataset)
data = pd.DataFrame({
    'input_text': [
        "summarize: Query: Predict patient readmission rates for seniors\nTarget: readmission\nConditions: {'age': (60, 120)}\nPredictions: {'1': 30, '0': 70}",
        "summarize: Query: Show sales trends for electronics\nTarget: revenue\nConditions: {'category': 'Electronics'}\nPredictions: {'mean': 1500, 'std': 200}"
    ],
    'target_text': [
        "The model predicts that 30% of senior patients are likely to be readmitted, while 70% are not.",
        "The average revenue for electronics is $1500 with a standard deviation of $200."
    ]
})

# Initialize tokenizer and model
tokenizer = T5Tokenizer.from_pretrained('t5-small')
model = TFT5ForConditionalGeneration.from_pretrained('t5-small')

# Tokenize data
def tokenize(texts, max_length=512):
    return tokenizer(
        texts.tolist(),
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors='tf'
    )

# Split data
X_train, X_val, y_train, y_val = train_test_split(
    data['input_text'], data['target_text'], test_size=0.2, random_state=42
)

# Tokenize inputs and targets
input_encodings = tokenize(X_train, max_length=512)
target_encodings = tokenize(y_train, max_length=128)

val_input_encodings = tokenize(X_val, max_length=512)
val_target_encodings = tokenize(y_val, max_length=128)

# Create TensorFlow datasets
train_dataset = tf.data.Dataset.from_tensor_slices((
    {'input_ids': input_encodings['input_ids'], 'attention_mask': input_encodings['attention_mask']},
    {'labels': target_encodings['input_ids']}
)).batch(8)

val_dataset = tf.data.Dataset.from_tensor_slices((
    {'input_ids': val_input_encodings['input_ids'], 'attention_mask': val_input_encodings['attention_mask']},
    {'labels': val_target_encodings['input_ids']}
)).batch(8)

# Compile model
optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5)
model.compile(optimizer=optimizer, loss=model.hf_compute_loss)

# Train model
model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=3
)

# Save fine-tuned model
model.save_pretrained('models/fine_tuned_t5')
tokenizer.save_pretrained('models/fine_tuned_t5')