import boto3

region = boto3.session.Session().region_name
account_id = boto3.client('sts').get_caller_identity().get('Account')
print("<region>: ", region)
print("<account_id>: ", account_id)
training_experiment_name="planradar-openclip-jan2025"
training_data_dir="planradar-openclip-jan2025-training-bucket" #bucket name where Jan 2025 data is stored or ../data for local
model_bucket_name=f"sagemaker-{region}-{account_id}"
model_folder_name_in_model_bucket="planradar-openclip-jan2025-model-bucket"
local_img_zip_path="data/images_small.zip"
local_description_path="data/descriptions_small.json"
batch_size = 2
num_epochs = 10
