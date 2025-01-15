'''

Inference script that runs in AWS Sagemaker endpoint
Receives image / text payload from local
Preprocess image & tokenize text for predicting image & text vectors
'''
import io
import os
import torch
from PIL import Image
import open_clip
import logging








class ClipInference(object):

    def __init__(self):

        self.initialized = False
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess_train, self.preprocess_val = open_clip.create_model_and_transforms("RN50")
        self.tokenizer = open_clip.get_tokenizer("RN50")
        self.logger = logging.getLogger("ConsoleLogger")
        self.logger.setLevel(logging.INFO)
        console_handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        self.logger.info('Model loaded')




    def initialize(self, context):

        self.initialized = True
        properties = context.system_properties
        ckpt_dir = properties.get("model_dir")
        ckpt_name=os.path.join(ckpt_dir,'best_model.chkpt')
        self.model.load_state_dict(state_dict=torch.load(ckpt_name)['model'])
        self.model.eval()
        self.logger.info('Model loaded with chkpt')

    def check_bytearray_type(self,data: bytearray):
        try:
            data.decode('utf-8')
            self.logger.info("payload is a string.")
            return "string"
        except UnicodeDecodeError:
            pass

        try:
            image = Image.open(io.BytesIO(data))
            image.verify()  # Verify the image file
            self.logger.info("payload is an image.")
            return "image"
        except Exception as e:
            self.logger.error('Unsupported payload input')
        return "unknown"


    def inference(self,request):

        payload=request[0]['body']
        self.logger.info('Checking payload type')
        type_of_payload=self.check_bytearray_type(payload)

        if type_of_payload=="image":
            self.logger.info('Predicting in image mode')
            data = Image.open(io.BytesIO(payload))
            data = self.preprocess_val(data).unsqueeze(0)
            self.logger.info('Input shape: '+str(data.shape))
            output=self.model.encode_image(data.to(self.device))
        elif type_of_payload == "string":
            self.logger.info('Predicting in text mode')
            data=payload.decode()
            data=self.tokenizer(data)
            self.logger.info('Input shape: ' + str(data.shape))
            output=self.model.encode_text(data.squeeze(1).to(self.device))
        else:
            self.logger.error('Incomplete Prediction')
            return []
        self.logger.info('Prediction completed')
        buffer = io.BytesIO()
        torch.save(output, buffer)
        tensor_bytes = buffer.getvalue()
        self.logger.info('Output ready')
        return [tensor_bytes]



    def handle(self, data, context):

        return self.inference(data)


_service = ClipInference()



def handle(data, context):
    if not _service.initialized:
        _service.initialize(context)

    if data is None:
        return None

    return _service.handle(data, context)


