import torch
import tarfile
import json
from PIL import Image
import os
import boto3
from config import region,account_id

class Dataset(torch.utils.data.Dataset):
    def __init__(self, descr_fpath, image_folder, preprocess, tokenizer):
        self.descriptions = list(json.load(open(descr_fpath, mode="r")).items())
        self.image_folder = image_folder
        self.preprocess = preprocess
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.descriptions)

    def __getitem__(self, idx):
        img_fname, description = self.descriptions[idx]
        img = Image.open(os.path.join(self.image_folder, img_fname))
        img = self.preprocess(img)
        tokens = self.tokenizer(description)
        return img, tokens

def zip_model_to_upload_to_s3(filepath):
    """
        Zip model as tar.gz tp upload to an S3 bucket.

        :param filename: modelpath to be zipped

    """
    source_files = [filepath]
    #output_zip_file = f"{filepath.split('/')[-1].strip('.chkpt')}.tar.gz"
    output_zip_file = filepath.replace('.chkpt','.tar.gz')

    if not os.path.isfile(output_zip_file):
        with tarfile.open(output_zip_file, mode="w:gz") as t:
            for sf in source_files:
                t.add(sf, arcname=os.path.basename(sf))
            print(f"ZIP file is created at {output_zip_file} location.")
        return True,output_zip_file
    else:
        print(f"ZIP file '{output_zip_file}' already exist. Please re-run the cell after removing it.")
        return False,None

def unzip_s3file(filepath,dest):
    os.system(f"unzip -q {filepath} -d {dest}")

def upload_file_to_s3(bucket_name, file_path, s3_key):
    """
    Upload a file to an S3 bucket.

    :param bucket_name: Name of the S3 bucket
    :param file_path: Path of the file to upload
    :param s3_key: Key (name) for the file in the S3 bucket
    """
    s3 = boto3.client('s3')

    try:
        s3.upload_file(file_path, bucket_name, s3_key)
        print(f"File uploaded successfully to {bucket_name}/{s3_key}")
    except Exception as e:
        print(f"Error uploading file: {e}")


def download_file_from_s3(bucket_name, download_path, s3_key):
    """
    Download a file from an S3 bucket.

    :param bucket_name: Name of the S3 bucket
    :param s3_key: Key (name) of the file in the S3 bucket
    :param download_path: Local path to save the downloaded file
    """
    s3 = boto3.client('s3')

    try:
        s3.download_file(bucket_name, s3_key, download_path)
        print(f"File downloaded successfully to {download_path}")
    except Exception as e:
        print(f"Error downloading file: {e}")


def create_s3_bucket(bucket_name):
    """
    Create an S3 bucket in a specified region.

    :param bucket_name: Name of the S3 bucket
    :param region: AWS region for the bucket (e.g., 'us-west-2')
    :return: True if bucket created successfully, otherwise False
    """
    try:
        # Create an S3 client
        s3_client = boto3.client('s3')
        s3_client.create_bucket(Bucket=bucket_name)
        print(f"Bucket '{bucket_name}' created successfully.")
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False


def send_email_via_sns(subject, message):
    """
    Send an email notification using Amazon SNS.

    :param topic_arn: The ARN of the SNS topic
    :param subject: Subject of the email
    :param message: Message content
    """
    topic_arn = f"arn:aws:sns:{region}:{account_id}:first_topic"

    try:
        sns_client = boto3.client('sns', region_name='us-east-1')
        # Publish a message to the SNS topic
        response = sns_client.publish(
            TopicArn=topic_arn,
            Subject=subject,
            Message=message
        )
        print(f"Message sent! Message ID: {response['MessageId']}")

    except Exception as e:
        print(f"An error occurred: {e}")





