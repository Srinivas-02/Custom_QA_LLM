# views.py
from pathlib import Path
from django.conf import settings
from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.response import Response
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

class QAView(APIView):

    def post(self, request):
        # Load the custom-trained model
        MODEL_PATH = Path(settings.BASE_DIR) / "custom_flan_t5_lora"
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH)

        question = request.data.get("question")
        if not question:
            return Response({"error": "No question provided"}, status=400)

        # Generate answer using the custom-trained model
        inputs = tokenizer(question, return_tensors="pt", truncation=True, max_length=512)
        outputs = model.generate(
            **inputs, 
            max_length=512,
            num_return_sequences=1,
            do_sample=True,  # Enable sampling
            temperature=0.7,  # Control randomness
            top_k=50,  # Top-k sampling
            top_p=0.95,  # Nucleus sampling
            no_repeat_ngram_size=2  # Prevent repetition
        )
        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

        return Response({"question": question, "answer": answer})
    

class HomeView(APIView):
    def get(self, request):
        # Render the QA template
        return render(request, 'index.html')