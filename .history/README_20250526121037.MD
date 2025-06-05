Multi-Domain ML System

A modular machine learning system supporting 21 domains (e.g., customer support, energy, marketing) with automated domain detection, data processing, model training, and query handling. Uses DistilBERT for intent classification and T5 for response generation.

Project Structure





src/





domain_configs/: Domain-specific configurations (customer_support.py, energy.py, etc.).



utils/: Utility modules (logger.py, data_processor.py, llm_manager.py, etc.).



main.py: Streamlit CLI application.



fine_tune_distilbert.py: Script to fine-tune DistilBERT for intent classification.



fine_tune_t5.py: Script to fine-tune T5 for response generation.



logs/: Log files for system and domain-specific activities.



models/: Trained models and fine-tuned LLMs.



data/: Sample datasets (e.g., intent_dataset.csv, response_dataset.csv).



requirements.txt: Project dependencies.

Setup





Clone the repository:

git clone <repository-url>
cd multi-domain-ml-system



Install dependencies:

pip install -r requirements.txt



Prepare datasets:





Place your dataset (CSV or Excel) in the data/ directory.



For fine-tuning, prepare:





intent_dataset.csv: Columns query, intent (e.g., "predict sales", "prediction").



response_dataset.csv: Columns context, response (e.g., "Query: predict sales", "Predicted sales: 1000").



Fine-tune models (optional):

python src/fine_tune_distilbert.py data/intent_dataset.csv
python src/fine_tune_t5.py data/response_dataset.csv

Usage





Run the application:

streamlit run src/main.py



Interact via Streamlit:





Upload a dataset (CSV or Excel).



View detected domain and data preview.



Train a model (optional).



Enter queries (e.g., "Predict energy consumption for solar", "Show performance metrics").



View results, including summaries and visualizations.

Domains Supported





Customer Support



Entertainment



Gaming



Legal



Marketing



Logistics



Manufacturing



Real Estate



Agriculture



Energy



Hospitality



Automobile



Telecommunications



Government



Food & Beverage



IT Services



Event Management



Insurance



Retail



HR Resources



Banking

Dependencies

See requirements.txt for a complete list. Key dependencies include:





Python 3.8+



pandas



numpy



plotly



streamlit



scikit-learn



transformers



torch



joblib

Notes





Ensure domain-specific query-response pairs are collected for fine-tuning T5 and DistilBERT.



The create_analysis method in domain configs is a placeholder; enhance with domain-specific visualizations.



Logs are stored in logs/ with timestamps and domain-specific files.



Models are saved in models/ with unique names.

Contributing

Contributions are welcome! Please open an issue or submit a pull request for bug fixes or enhancements.

License

MIT License