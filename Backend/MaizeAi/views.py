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


    #fs = FileSystemStorage(location='MaizeAi/RCNN/input')  # adjust the location as necessary
    #file_path = fs.save(filename, f)
    #full_path = fs.path(file_path)
    #print("\nfilename: ", filename)
    
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

# View to handle image uploads from the frontend
@csrf_exempt  
def image_upload_view(request):
    if request.method == 'POST' and request.FILES['file']:
        # Save the uploaded image and call the ML script
        print(request)
        print(request.FILES['file'])
        image = request.FILES['file']
        print(type(image))
        try:
            image_path, original_base_name, unique_identifier = handle_uploaded_file(image, image.name)
            
            # Assuming the ML script saves the processed image as 'processed.jpg' in the output directory
            #image_with_boxes = os.path.join(output_directory, 'processed.jpg')
            
            # Generate the response data
            print("\npath: ", image_path)

            count_filepath = os.path.join('MaizeAi', 'RCNN', 'output', 'Count', f'{original_base_name}_{unique_identifier}.txt')
            with open(count_filepath, 'r') as f:
                count = f.read()
            
            response_data = {
                'tassel_count': count,
                # Add any other data to return to the frontend 
            }
            print(count)
            return JsonResponse({'tassel_count': count})
        except Exception as e:
            import traceback
            traceback.print_exc()
            return JsonResponse({'error': str(e)}, status=500)

    return JsonResponse({'error': 'No image provided'}, status=400)