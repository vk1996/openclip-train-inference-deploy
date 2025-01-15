import torch
print(torch.__version__)
import open_clip
import mlflow
from torchinfo import summary
from utils import Dataset,zip_model_to_upload_to_s3,unzip_s3file,download_file_from_s3,upload_file_to_s3,create_s3_bucket,send_email_via_sns
from config import training_experiment_name,training_data_dir,model_bucket_name,local_img_zip_path,local_description_path,batch_size,num_epochs,model_folder_name_in_model_bucket
import boto3




def save_model(name):
    checkpoint = {
    "model": model.state_dict(),
    "optimizer": optimizer.state_dict(),
    "scaler": scaler.state_dict(),
    "scheduler": scheduler.state_dict(),
    "epoch": epoch,
    }
    torch.save(checkpoint, f"models/{name}.chkpt")


try :


    ############# Download client images & description from S3 #################
    download_file_from_s3(training_data_dir,local_img_zip_path,local_img_zip_path.split('/')[-1])
    unzip_s3file(local_img_zip_path,'data')
    download_file_from_s3(training_data_dir, local_description_path, local_description_path.split('/')[-1])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms("RN50", "cc12m")
    model.transformer.eval()
    tokenizer = open_clip.get_tokenizer("RN50")
    dataset = Dataset(local_description_path, local_img_zip_path.replace('.zip',''),preprocess_train, tokenizer)
    dataset_train, dataset_test = torch.utils.data.random_split(dataset, [0.7, 0.3])
    dataloader_train = torch.utils.data.DataLoader(
        dataset_train,
        shuffle=True,
        batch_size=batch_size,
        num_workers=4,
        drop_last=True,
    )

    dataloader_test = torch.utils.data.DataLoader(
        dataset_test,
        shuffle=False,
        batch_size=batch_size,
        num_workers=4,
        drop_last=True,
    )


    optimizer = torch.optim.AdamW(model.visual.parameters())
    loss_fn = torch.nn.CrossEntropyLoss()
    model = model.to(device)
    scaler = torch.amp.GradScaler(device)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=0.0001,
        steps_per_epoch=len(dataloader_train),
        epochs=num_epochs,
        pct_start=0.1,
    )

    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment(training_experiment_name)

    ################## Logging & Monitoring model & system metrics with mlflow #############
    with mlflow.start_run(log_system_metrics=True):
        mlflow.set_tag("Training Info", training_experiment_name)#
        curr_lr = scheduler.get_last_lr()
        mlflow.log_param("starting_lr", curr_lr)
        mlflow.log_param("training data version", training_data_dir)
        mlflow.log_param("batch_size",batch_size)
        print("starting_lr:", curr_lr)
        mlflow.log_param("num_epochs", num_epochs)

        with open(f"{training_experiment_name}_model_summary.txt", "w") as f:
            f.write(str(summary(model)))
        mlflow.log_artifact(f"{training_experiment_name}_model_summary.txt")

        for epoch in range(num_epochs):

            curr_lr = scheduler.get_last_lr()
            mlflow.log_metric("lr", curr_lr[0], epoch + 1)
            print("last_lr:", curr_lr)

            print("epoch", epoch + 1)
            model.visual.train()

            losses = []
            for img, tokens in dataloader_train:
                with torch.autocast(device, dtype=torch.float16):
                    # encode features
                    img_features = model.encode_image(img.to(device))
                    text_features = model.encode_text(tokens.squeeze(1).to(device))

                    # normalize features
                    img_features = img_features / img_features.norm(dim=-1, keepdim=True)
                    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

                    # get loss
                    logits = 100.0 * img_features @ text_features.T
                    loss = loss_fn(logits, torch.arange(batch_size).to(device))


                # do optimization step
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                losses.append(loss.item())

            train_loss = sum(losses) / len(losses)
            print("train loss:", train_loss)
            mlflow.log_metric("train_loss", train_loss, epoch + 1)

            losses = []
            model.visual.eval()
            with torch.no_grad(), torch.autocast(device, dtype=torch.float16):
                img_features = model.encode_image(img.to(device))
                text_features = model.encode_text(tokens.squeeze(1).to(device))

                img_features = img_features / img_features.norm(dim=-1, keepdim=True)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)

                logits = 100.0 * img_features @ text_features.T
                loss = loss_fn(logits, torch.arange(batch_size).to(device))
                losses.append(loss.item())

            val_loss = sum(losses) / len(losses)
            print("val loss:", val_loss)
            mlflow.log_metric("val_loss", val_loss, epoch + 1)
            print("test:", val_loss)
            print()
            ############# Model versioning with save best callbacks ##############
            if epoch > 0:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    save_model('best_model')
            else:
                save_model('best_model')
                best_val_loss = val_loss
            save_model('latest_model')
            # mlflow.pytorch.log_model(model, f"models/{training_experiment_name}")
        ############# Model Storage with S3 #################
        status,zip_path=zip_model_to_upload_to_s3('models/best_model.chkpt')
        create_s3_bucket(model_bucket_name)
        upload_file_to_s3(model_bucket_name,zip_path,f"{model_folder_name_in_model_bucket}/{zip_path.split('/')[-1]}")
        ################### Notification & Warning with Amazon SNS email publish ##################
        message=f"{training_experiment_name} training Completed Successfully"
        send_email_via_sns("[Success] PlanRadar OpenClip Training AWS SNS", message)

except Exception as e:
    print(e)
    message=f"{training_experiment_name} training interrupted by following exception \n {str(e)}"
    send_email_via_sns("[Error] PlanRadar OpenClip Training AWS SNS", message)