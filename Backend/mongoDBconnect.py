import pymongo
import boto3

# mongo
url = 'mongodb+srv://Brenda:hellogrillsGgw@cluster0.ruwnyh7.mongodb.net/'
client = pymongo.MongoClient(url)
db = client['MaizeAi']

# aws s3
aws_access_key_id = 'AKIASJ5W2RICQL3OAZ45'
aws_secret_access_key = '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'
s3_bucket_name = 'maize-ai'
s3_client = boto3.client('s3', aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key)