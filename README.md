# Cutsom_QA_LLM
Fine-tune an open-source LLM (FLAN-T5) on custom data and deploy it using Django. Includes a web interface for real-time Q&amp;A.

Custom QA LLM with Flan-T5 and Django
🚀 Project Overview
Fine-tune an open-source LLM (FLAN-T5) on custom data and deploy it using Django, with a web interface for real-time Q&A interactions.

📋 Prerequisites
Python 3.10+
pip
Django
Transformers
PyTorch

🛠️ Setup & Installation
1. Clone Repository
    git clone < repo_link >
    cd custom_llm_qa
2. Install Dependencies
    pip install -r requirements.txt

🤖 Model Training
    Prepare Training Data
    Create custom_data.txt:
        What is AI? --- Artificial Intelligence is...
        How does machine learning work? --- Machine learning involves...
    Train Custom Flan-T5
    python model_training.py
    Move the reuslt to the django folder.
    
    custom_llm_qa/
    ├── custom_flan_t5/  # Place the model here
    │   ├── pytorch_model.bin
    │   ├── config.json
    │   └── ...
    ├── templates/
    │   └── qa_app/
    │       └── index.html
    └── custom_llm_qa/
        └── qa_app/
            └── views.py

🌐 Django Deployment
    Run Migrations
        python manage.py migrate
    Start Server
        python manage.py runserver

🖥️ Access Application

Open browser
Navigate to http://127.0.0.1:8000/qa_app/ask/

🔍 Interaction

Enter question in text area
Click "Get Answer"
View AI-generated response

🛠️ Troubleshooting

Verify dependencies
Check custom_data.txt format
Confirm model path in views.py

📈 Performance Optimization

Adjust batch size/epochs in model_training.py
Experiment with training data
Monitor model performance