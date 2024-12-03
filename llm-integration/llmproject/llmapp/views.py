from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
from . import clip_vectorization
import json
from django.views.decorators.csrf import csrf_exempt
# from openai import OpenAI
from groq import Groq
from django.conf import settings
import os
# Create your views here.

def home(request):
    return render(request,'home.html',{"message" : "Welcome to your page!!"})

openai_api_key = "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"

groq_api_key = "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"

client = Groq(api_key = groq_api_key)

@csrf_exempt
def getDataFromOpenAIAPI(request):
    if request.method == "POST":
        try:
            data = json.loads(request.body)
            query = data.get("query", "")
            completion = client.chat.completions.create(
                    model="llama3-8b-8192",
                    messages=[
                        {
                            "role": "user",
                            "content": query
                        }
                    ],
                    temperature=1,
                    max_tokens=1024,
                    top_p=1,
                    stream=True,
                    stop=None,
                )
            ans = ""
            for chunk in completion:
                ans += chunk.choices[0].delta.content or ""
            return JsonResponse({'response': ans})
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)
    return JsonResponse({'error': 'Invalid request method.'}, status=400)

@csrf_exempt
def uploadFile(request):
    if request.method == 'POST' and request.FILES.get('image'):
        # Retrieve the uploaded file
        uploaded_file = request.FILES['image']
        # Define the local path to save the image
        save_dir = os.path.join(settings.MEDIA_ROOT, 'images')
        os.makedirs(save_dir, exist_ok=True)  # Create the directory if it doesn't exist
        file_path = os.path.join(save_dir, uploaded_file.name)

        try:
            with open(file_path, 'wb+') as destination:
                for chunk in uploaded_file.chunks():
                    destination.write(chunk)

            return JsonResponse({'message': 'Image uploaded successfully', 'file_path': file_path})
        except Exception as e:
            return JsonResponse({'error': f'Failed to save image: {str(e)}'}, status=500)
    else:
        return JsonResponse({'error': 'No image file provided'}, status=400)
    
@csrf_exempt
def getEmbedding(request):
    if request.method == 'POST':
        try:
            textEmbedding = []
            imageEmbedding = []
            data = json.loads(request.body)
            type = data.get("type", "")
            print(type)
            if type == "text":
                text = data.get("text","")
                textEmbedding = clip_vectorization.vectorize_text(text)
                return JsonResponse({'response': textEmbedding})
            else:
                imagePath = data.get("imageFilePath","")
                imageEmbedding = clip_vectorization.vectorize_image(imagePath)
                return JsonResponse({'response': imageEmbedding})
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)
    return JsonResponse({'error': 'Invalid request method.'}, status=400)