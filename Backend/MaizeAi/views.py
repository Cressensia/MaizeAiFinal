from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.core.files.storage import default_storage
from django.core.files.storage import FileSystemStorage
from django.contrib.auth.decorators import login_required
import subprocess
from .models import MaizeCounter
import uuid
import base64
import os

from django.contrib.auth.decorators import login_required
from django.contrib.auth import get_user_model
from bson.json_util import dumps


import boto3
from datetime import datetime
from pymongo import MongoClient
from botocore.exceptions import NoCredentialsError

from jose import jwt, jwk, JWTError
import requests

client = MongoClient('mongodb+srv://cressensia:hellogrillsGgw@cluster0.ruwnyh7.mongodb.net/')
db = client['MaizeAi']
collection = db['counter']
bucket = "maize-ai"

def upload_to_s3(local_path, s3_bucket, s3_key):
    try:
        # Initialize S3 client
        s3 = boto3.client('s3', aws_access_key_id='AKIASJ5W2RIC5UTSVCQ7', aws_secret_access_key='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')
        s3.upload_file(local_path, s3_bucket, s3_key)
        print(f"file uploaded to s3: s3://{s3_bucket}/{s3_key}")
    except NoCredentialsError:
        print("credentials not found")

# View to handle image uploads from the frontend
@csrf_exempt
def image_upload_view(request):
    print("Authorization Header:", request.headers.get('Authorization'))
    print("Email:", request.POST.get('userEmail'))

    # check authtoken for login session
    auth_token = request.headers.get('Authorization')
    if not auth_token:
        return JsonResponse({'error': 'Authentication token not provided'}, status=401)
    
    # get user email to create unique s3 directory
    user_email = request.POST.get('userEmail')
    if not user_email:
        return JsonResponse({'error': 'Email not provided'}, status=400)
    
    upload_date = request.POST.get('uploadDate')

    if request.method == 'POST' and request.FILES.getlist('files'):
        response_data = {'results': [], 'total_count': 0}

        for image in request.FILES.getlist('files'):
            try:
                # Save the uploaded image and call the ML script
                fs = FileSystemStorage(location='MaizeAi/RCNN/input')
                unique_identifier = str(uuid.uuid4())[:8]
                base_name, extension = os.path.splitext(image.name)
                original_base_name, _ = base_name.rsplit('_', 1) if '_' in base_name else (base_name, 'temp')
                new_filename = f"{original_base_name}_{unique_identifier}{extension}"

                file_path = fs.save(new_filename, image)
                full_path = fs.path(file_path)

                s3_input_key = f'{user_email}/input/{new_filename}'
                #s3_input_key = f'input/{new_filename}'
                upload_to_s3(full_path, bucket, s3_input_key)

                output_directory = 'MaizeAi/RCNN/output'

                result = subprocess.run(
                    ['python', 'MaizeAi/RCNN/infer1.py', full_path, output_directory],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
                
                if result.returncode != 0:
                    raise Exception(result.stderr)

                count_filepath = os.path.join(output_directory, 'Count', f'{original_base_name}_{unique_identifier}.txt')
                with open(count_filepath, 'r') as f:
                    count = f.read()

                output_filepath = os.path.join(output_directory, 'detection', f'{original_base_name}_{unique_identifier}_with_boxes.jpg')
                s3_output_key = f'{user_email}/output/detection/{original_base_name}_{unique_identifier}_with_boxes.jpg'
                #s3_output_key = f'output/detection/{original_base_name}_{unique_identifier}_with_boxes.jpg'
                upload_to_s3(output_filepath, bucket, s3_output_key)
                
                with open(output_filepath, 'rb') as f:
                    processed_image_binary_data = f.read()
                    image_data = base64.b64encode(processed_image_binary_data).decode('utf-8')

                # Insert into MongoDB
                maize_document = MaizeCounter.insert_one({
                    # 'user_id': request.user.id,
                    'original_image': f's3://{bucket}/{s3_input_key}', #save s3 uri instead of key
                    'processed_image': f's3://{bucket}/{s3_output_key}',
                    'tassel_count': int(count),
                    'upload_date': upload_date,
                    'user_email':  user_email,
                })

                # Append result to the response data
                response_data['results'].append({
                    'tassel_count': count,
                    'image_data': image_data,
                    'document_id': str(maize_document.inserted_id),
                    'upload_date': upload_date,
                    'user_email':  user_email,
                })
                response_data['total_count'] += int(count)
                
            except Exception as e:
                return JsonResponse({'error': str(e)}, status=500)

        return JsonResponse(response_data)
    else:
        return JsonResponse({'error': 'No image provided'}, status=400)
    

@login_required
def get_user_results(request):
    # Fetch results from MongoDB that match the logged-in user's ID
    user_results = MaizeCounter.find({'user': request.user.id})

    transformed_results = []
    for result in user_results:
        transformed_result = {
            'original_image': result['original_image'],
            'processed_image': result['processed_image'],
            'tassel_count': result['tassel_count'],
        }
        transformed_results.append(transformed_result)

    # Serialize the MongoDB cursor to JSON
    user_results_json = dumps(user_results)
    
    return HttpResponse(user_results_json, content_type='application/json')

# get email from user
# @login_required
def get_user_email(request):
    if request == "GET":  # and request.user.is_authenticated:
        return JsonResponse()

    
import os
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST
from django.core.files.storage import FileSystemStorage
# from MaizeAi.RCNN.inferDD import process_image_and_generate_prediction, category_to_disease
from MaizeAi.RCNN.inferDD import process_image_and_generate_prediction, model

@csrf_exempt  # Disable CSRF protection for this view (for development purposes only)
@require_POST  # Allow only POST requests to this view
def upload_imageMaizeDisease(request):
    if request.method == "POST" and request.FILES.get("image"):
        # Get the uploaded image from the request
        image_file = request.FILES["image"]

        # Use FileSystemStorage for handling file uploads
        fs = FileSystemStorage()

        # Generate a unique filename for the uploaded image file
        filename = fs.save(f"inputDD/{image_file.name}", image_file)
        uploaded_file_url = fs.url(filename)
        temp_image_path = fs.path(filename)

        # Process the uploaded image and generate predictions
        output_path, disease_name = process_image_and_generate_prediction(temp_image_path)

        # Remove the input image after processing
        fs.delete(filename)

        # Prepare the path to be relative to the 'outputDD' directory
        relative_output_path = os.path.relpath(output_path, 'MaizeAi/RCNN/outputDD')

        # Return the prediction and disease type as JSON response
        return JsonResponse({
            "message": "Image processed successfully",
            "disease_type": disease_name,
            "image_path": relative_output_path,
            "uploaded_image_url": uploaded_file_url  #this one no need one actually sfksdhjvfhkhs
        })

    return JsonResponse({"error": "Invalid request"}, status=200)