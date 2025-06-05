import spacy
from transformers import pipeline

# Load NLP models once
nlp = spacy.load("en_core_web_sm")
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

def summarize_text(text):
    doc = nlp(text)
    sentences = list(doc.sents)
    summary = ' '.join([str(s) for s in sentences[:3]])
    return summary

def extract_keywords(text):
    doc = nlp(text)
    return list(set(chunk.text.lower() for chunk in doc.noun_chunks if len(chunk.text) > 3))

def extract_named_entities(text):
    doc = nlp(text)
    return [{'text': ent.text, 'label': ent.label_} for ent in doc.ents]

def get_summary_with_transformer(text, max_length=100, min_length=30):
    return summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)[0]['summary_text']
