from django.shortcuts import render
from django.http import HttpResponse
import os
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.core.files.storage import default_storage
# from MaizeAi.RCNN.infer1 import process_images_and_predict
from django.core.files.storage import FileSystemStorage
import subprocess
from django.core.files.storage import default_storage

import uuid
import base64

# Create your views here.

# Function to handle the uploaded file and call the ML script
def handle_uploaded_file(f, filename):
    fs = FileSystemStorage(location='MaizeAi/RCNN/input')  # adjust the location as necessary
    # generate unique id
    unique_identifier = str(uuid.uuid4())[:8]

    base_name, extension = os.path.splitext(filename)

    # remove appended random string
    original_base_name, random_text = base_name.rsplit('_', 1)

    # set new filename - basename + unique id
    new_filename = f"{original_base_name}_{unique_identifier}{extension}"

    # save as new filename 
    file_path = fs.save(new_filename, f)
    full_path = fs.path(file_path)
    print("\nfilename: ", new_filename)

    # Directory to store the output
    '''
    output_directory = os.path.join('MaizeAi/RCNN/output', filename.split('.')[0])  # Creates a unique directory for each upload
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    '''
    output_directory = os.path.join('MaizeAi/RCNN/output')

    # Call the ML script using subprocess
    result = subprocess.run(
        ['python', 'MaizeAi/RCNN/infer1.py', full_path, output_directory],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    # Check if the script was executed successfully
    if result.returncode != 0:
        raise Exception(result.stderr)
    
    return full_path, original_base_name, unique_identifier



@csrf_exempt
def image_upload_view(request):
    print("Request Method:", request.method)
    print("Request Files:", request.FILES)
    if request.method == 'POST' and request.FILES.getlist('files'):
        # Save the uploaded image and call the ML script
        print(request)
        print(request.FILES.getlist('files')) # update to list
        images = request.FILES.getlist('files')

        try:
            response_data = {'results': [], 'total_count': 0}

            for image in images:
                image_path, original_base_name, unique_identifier = handle_uploaded_file(image, image.name)
                print(f"processing {image.name}")

                count_filepath = os.path.join('MaizeAi', 'RCNN', 'output', 'Count', f'{original_base_name}_{unique_identifier}.txt')
                with open(count_filepath, 'r') as f:
                    count = f.read()
                    print(f"{original_base_name}_{unique_identifier}.jpg : {count}")

                output_filepath = os.path.join('MaizeAi', 'RCNN', 'output', 'detection', f'{original_base_name}_{unique_identifier}_with_boxes.jpg')
                print(f"output path: {output_filepath}")

                with open(output_filepath, 'rb') as f:
                    image_data = base64.b64encode(f.read()).decode('utf-8')

                response_data['results'].append({
                    'tassel_count': count,
                    'image_data': image_data
                    # Add any other data to return to the frontend
                })

                response_data['total_count'] += int(count)
                print(response_data['results'][-1]['tassel_count'])

            return JsonResponse(response_data)
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)

    return JsonResponse({'error': 'No image provided'}, status=400)