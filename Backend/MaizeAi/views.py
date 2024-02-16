import json
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.core.files.storage import FileSystemStorage
import subprocess
import uuid
import base64
import os
import glob
from MaizeAi.RCNN.inferDD import process_image_and_generate_prediction
import boto3
from pymongo import MongoClient
from botocore.exceptions import NoCredentialsError
from bson import ObjectId
from urllib.parse import quote_plus, unquote_plus, urlparse
from json import loads
from datetime import datetime

from dotenv import load_dotenv

load_dotenv()

aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
mongodb_uri = os.getenv("MONGODB_URI")

mongo_client = MongoClient(mongodb_uri)

import logging
logger = logging.getLogger(__name__)

db = mongo_client['MaizeAi']
counter_collection = db['counter']
disease_collection = db['disease']
user_collection = db['users']

bucket = "maize-ai"

def upload_to_s3(local_path, s3_bucket, s3_key):
    try:
        # Initialize S3 client
        s3 = boto3.client('s3', aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key)
        print(f"uploading file: {local_path} to s3://{s3_bucket}/{s3_key}")
        s3.upload_file(local_path, s3_bucket, s3_key)
        print(f"file uploaded to s3: s3://{s3_bucket}/{s3_key}")
    except NoCredentialsError:
        print("credentials not found")

def delete_from_s3(s3_bucket, key):
    s3 = boto3.client('s3', aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key)
    s3.delete_object(Bucket=s3_bucket, Key=key)

def get_s3_url(s3_bucket, key, region='ap-southeast-1'):
    parts = key.split('/')
    encoded_parts = [quote_plus(part) for part in parts] #encode special characters eg '@'
    encoded_key = '/'.join(encoded_parts)
    return f"https://{s3_bucket}.s3.{region}.amazonaws.com/{encoded_key}"

def reverse_s3_url(encoded_url):
    parsed_url = urlparse(encoded_url)
    s3_key = unquote_plus(parsed_url.path).lstrip('/')
    return s3_key


# View to handle image uploads from the frontend
@csrf_exempt
def image_upload_view(request):
    print("Authorization Header:", request.headers.get("Authorization"))
    print("Email:", request.POST.get("userEmail"))

    # check authtoken for login session
    auth_token = request.headers.get("Authorization")
    if not auth_token:
        return JsonResponse({"error": "Authentication token not provided"}, status=401)

    # get user email to create unique s3 directory
    user_email = request.POST.get("userEmail")
    if not user_email:
        return JsonResponse({"error": "Email not provided"}, status=400)

    upload_date = request.POST.get("uploadDate")

    if request.method == "POST" and request.FILES.getlist("files"):
        response_data = {"results": [], "total_count": 0}

        for image in request.FILES.getlist("files"):
            try:
                # Save the uploaded image and call the ML script
                fs = FileSystemStorage(location="MaizeAi/RCNN/input")
                unique_identifier = str(uuid.uuid4())[:8]
                base_name, extension = os.path.splitext(image.name)
                original_base_name, _ = (
                    base_name.rsplit("_", 1)
                    if "_" in base_name
                    else (base_name, "temp")
                )
                new_filename = f"{original_base_name}_{unique_identifier}{extension}"

                file_path = fs.save(new_filename, image)
                full_path = fs.path(file_path)

                s3_input_key = f"{user_email}/input/{new_filename}"
                # s3_input_key = f'input/{new_filename}'
                upload_to_s3(full_path, bucket, s3_input_key)

                output_directory = "MaizeAi/RCNN/output"

                result = subprocess.run(
                    ["python", "MaizeAi/RCNN/inferPheno.py", full_path, output_directory, user_email],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )

                if result.returncode != 0:
                    raise Exception(result.stderr)

                count_filepath = os.path.join(output_directory, "Count", f"{original_base_name}_{unique_identifier}.txt")
                with open(count_filepath, "r") as f:
                    count = f.read()

                # check each image if outlier files exists
                json_pattern = os.path.join(output_directory, "Outliers", f"{original_base_name}_{unique_identifier}*.json")
                json_files = glob.glob(json_pattern)

                # initialize array for all outliers
                aggregated_outliers = []

                for json_file in json_files:
                    try:
                        with open(json_file, 'r') as f:
                            data = json.load(f)
                            aggregated_outliers.append(data)
                    except Exception as e:
                        print(f"Error reading {json_file}: {e}")

                outliers = aggregated_outliers if aggregated_outliers else None

                # Retrieve detection result 
                output_filepath = os.path.join(
                    output_directory,
                    "detection",
                    f"{original_base_name}_{unique_identifier}_with_boxes.jpg",
                )
                s3_output_key = f"{user_email}/output/{original_base_name}_{unique_identifier}_with_boxes.jpg"
                upload_to_s3(output_filepath, bucket, s3_output_key)

                original_url = get_s3_url(bucket, s3_input_key)
                processed_url = get_s3_url(bucket, s3_output_key)

                with open(output_filepath, "rb") as f:
                    processed_image_binary_data = f.read()
                    image_data = base64.b64encode(processed_image_binary_data).decode("utf-8")
                
                # Insert into MongoDB
                maize_document = counter_collection.insert_one(
                    {
                        "original_image": original_url,  # save s3 uri instead of key
                        "processed_image": processed_url,
                        "tassel_count": int(count),
                        "upload_date": upload_date,
                        "user_email": user_email,
                        "outliers": outliers,
                    }
                )

                print(outliers)

                # Append result to the response data
                response_data["results"].append(
                    {
                        "tassel_count": count,
                        "image_data": image_data,
                        "document_id": str(maize_document.inserted_id),
                    }
                )
                response_data["total_count"] += int(count)

            except Exception as e:
                logger.error(str(e))
                return JsonResponse({"error": str(e)}, status=500)

        return JsonResponse(response_data)
    else:
        return JsonResponse({"error": "No image provided"}, status=400)


@csrf_exempt  # Disable CSRF protection for this view (for development purposes only)
def upload_imageMaizeDisease(request):
    if request.method == "POST" and request.FILES.get("image"):
        print("Email:", request.POST.get('userEmail'))
        
        # get user email to create unique s3 directory
        user_email = request.POST.get('userEmail')
        if not user_email:
            return JsonResponse({'error': 'Email not provided'}, status=400)
        
        upload_date = request.POST.get('uploadDate')

        # Get the uploaded image from the request
        image_file = request.FILES["image"]

        # Use FileSystemStorage for handling file uploads
        fs = FileSystemStorage()

        unique_identifier = str(uuid.uuid4())[:8]
        base_name, extension = os.path.splitext(image_file.name)
        new_filename = f"{base_name}_{unique_identifier}{extension}"

        file_path = fs.save(new_filename, image_file)
        temp_image_path = fs.path(file_path)

        s3_input_key = f'{user_email}/maize-disease/input/{new_filename}'
        upload_to_s3(temp_image_path, bucket, s3_input_key)
        print(s3_input_key)

        # Process the uploaded image and generate predictions
        output_path, disease_name = process_image_and_generate_prediction(temp_image_path)
 
        # split output_path into dir and filename 
        output_directory, output_filename = os.path.split(output_path)

        # join new filename
        full_path = os.path.join(output_directory, f'prediction_{base_name}_{unique_identifier}.jpg')

        print(output_path)
        print(full_path)

        s3_output_key = f'{user_email}/maize-disease/output/prediction_{new_filename}'
        upload_to_s3(full_path, bucket, s3_output_key)
        print(s3_output_key)

        # Remove the input image after processing
        fs.delete(new_filename)

        # Prepare the path to be relative to the 'outputDD' directory
        relative_output_path = os.path.relpath(output_path, 'MaizeAi/RCNN/outputDD')

        original_url = get_s3_url(bucket, s3_input_key)
        processed_url = get_s3_url(bucket, s3_output_key)

        # Insert into MongoDB
        maize_document = disease_collection.insert_one({
            'original_image': original_url, #save s3 url
            'processed_image': processed_url,
            'disease_type': disease_name,
            'upload_date': upload_date,
            'user_email':  user_email,
        })

        # Return the prediction and disease type as JSON response
        return JsonResponse({
            "message": "Image processed successfully",
            "disease_type": disease_name,
            "image_path": relative_output_path,
        })

    return JsonResponse({"error": "Invalid request"}, status=200)

@csrf_exempt
def save_user(request):
    if request.method == 'POST':
        email = request.POST.get('email')
        name = request.POST.get('name')
        print(email, name)

        # check if exist (registered but not verified)
        if user_collection.find_one({'email': email}):
            return JsonResponse({'error': 'Email already exists'}, status=400)

        user_collection.insert_one({
            'email': email,
            'name': name,
        })

        return JsonResponse({'message': 'User saved successfully'})

    return JsonResponse({'error': 'Invalid request'}, status=400)

@csrf_exempt
def manage_user(request):
    if request.method == 'GET':
        email = request.GET.get('email')
        user_details = user_collection.find_one({'email': email})
        print(user_details)
        if user_details:
            user_details['_id'] = str(user_details['_id'])
            return JsonResponse(user_details, status=200)
        else:
            return JsonResponse({'error': 'User not found'}, status=404)

    elif request.method == 'POST':
        # get new name
        email = request.POST.get('email')
        new_name = request.POST.get('name')
        print(email, new_name)

        # update 
        result = user_collection.update_one({'email': email}, {'$set': {'name': new_name}})

        if result.modified_count > 0:
            return JsonResponse({'message': 'User details updated successfully'}, status=200)
        else:
            return JsonResponse({'error': 'User details not updated'}, status=400)

    return JsonResponse({'error': 'Invalid request'}, status=400)


@csrf_exempt
def get_results_by_email(request):
    if request.method == 'GET':
        user_email = request.GET.get('user_email')

        if not user_email:
            return JsonResponse({'error': 'email not provided'}, status=400)

        counter_results = counter_collection.find({'user_email': user_email})
        disease_results = disease_collection.find({'user_email': user_email})

        counter_response_data = {
            'counter_results': [{
                'original_image': result['original_image'],
                'processed_image': result['processed_image'],
                'tassel_count': result['tassel_count'],
                'upload_date': result['upload_date'],
                'document_id': str(result['_id']),
                'plot_name': result.get('plot_name', ''),
                'section': result.get('section', ''),
                'outliers': result['outliers'],
            } for result in counter_results]
        }

        disease_response_data = {
            'disease_results': [{
                'original_image': result['original_image'],
                'processed_image': result['processed_image'],
                'disease_type': result['disease_type'],
                'upload_date': result['upload_date'],
                'document_id': str(result['_id']),
            } for result in disease_results]
        }
        
        response_data = {**counter_response_data, **disease_response_data}
        return JsonResponse(response_data)
    else:
        return JsonResponse({'error': 'Invalid request'}, status=405)
    

@csrf_exempt
def delete_record(request, collection_name, document_id):
    if request.method == 'DELETE':
        try:
            if collection_name == 'counter':
                collection = counter_collection
            elif collection_name == 'disease':
                collection = disease_collection
            else:
                return JsonResponse({'error': 'Invalid collection name'}, status=400)

            # get record from mongo
            record = collection.find_one({'_id': ObjectId(document_id)})
            print(record)

            original_encoded_url = record['original_image']
            original_s3_key = reverse_s3_url(original_encoded_url)
            processed_encoded_url = record['processed_image']
            processed_s3_key = reverse_s3_url(processed_encoded_url)

            delete_from_s3(bucket, original_s3_key)
            delete_from_s3(bucket, processed_s3_key)

            # delete from mongo
            collection.delete_one({'_id': ObjectId(document_id)})

            return JsonResponse({'message': 'deleted successfully'})
        except Exception as e:
            logger.error(str(e))
            return JsonResponse({'error': str(e)}, status=500)
    else:
        return JsonResponse({'error': 'Invalid request'}, status=405)

@csrf_exempt
def update_plots(request):
    print(request.method)
    if request.method == 'POST':
        try:
            document_id = request.POST.get('documentId')
            plot_name = request.POST.get('plotName')
            section = request.POST.get('section')
            print(f"Received document_id: {document_id}")

            document = counter_collection.find_one({'_id': ObjectId(document_id)})
            print(f"Found document: {document}")

            if document:  
                document['plot_name'] = plot_name
                document['section'] = section
     
                # update in mongo
                counter_collection.update_one({'_id': ObjectId(document_id)}, {'$set': document})

                updated_results_json = get_results_by_email(request).content
                updated_results = loads(updated_results_json.decode('utf-8'))
                return JsonResponse({'updatedResults': updated_results})
            else:
                return JsonResponse({'error': 'Document not found'}, status=404)

        except Exception as e:
            logger.error(str(e))
            return JsonResponse({'error': str(e)}, status=500)

    return JsonResponse({'error': 'Invalid request'}, status=400)


@csrf_exempt
def get_total_count(request):
    if request.method == 'GET':
        user_email = request.GET.get('user_email')

        if not user_email:
            return JsonResponse({'error': 'email not provided'}, status=400)

        # get all result of user
        results = counter_collection.find({'user_email': user_email})

        # sum all counts
        total_count = sum(result['tassel_count'] for result in results)

        return JsonResponse({'total_count': total_count})
    else:
        return JsonResponse({'error': 'Invalid request'}, status=405)
    

@csrf_exempt
def get_monthly_count(request):
    if request.method == 'GET':
        user_email = request.GET.get('user_email')

        if not user_email:
            return JsonResponse({'error': 'email not provided'}, status=400)

        # get all result of user
        results = counter_collection.find({'user_email': user_email})

        # initialize array for monthly counts
        monthly_count = [0] * 12

        # calculate monthly counts for current year
        current_year = datetime.now().year
        for result in results:
            upload_date = datetime.strptime(result['upload_date'], "%d/%m/%Y")
            if upload_date.year == current_year:
                month_index = upload_date.month - 1 # index 0 jan
                monthly_count[month_index] += result['tassel_count']
        print(monthly_count) # [0, 14, 0, 0, 16, 0, 0, 0, 0, 0, 0, 0]

        return JsonResponse({'monthly_count': monthly_count})
    else:
        return JsonResponse({'error': 'Invalid request'}, status=405)
