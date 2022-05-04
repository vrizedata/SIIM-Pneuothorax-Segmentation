import torch.onnx 
import torch,yaml
from model import ResUnet
from torch import onnx


with open('config.yaml') as yaml_file:
    config = yaml.safe_load(yaml_file)


model_path = config['CONFIGS']['PYTORCH_MODEL_PATH']
img_size = config['CONFIGS']['IMG_SIZE']
device = config['CONFIGS']['DEVICE']


model = ResUnet(input_channels=3,classes=1,encoder_weights=False)

def load_ckp(checkpoint_fpath, model, device='cpu'):
    """
    checkpoint_path: path to save checkpoint
    model: model that we want to load checkpoint parameters into       
    optimizer: optimizer we defined in previous training
    """
    # load check point
    if device == 'gpu':
      model = model.to('cuda:0')
      checkpoint = torch.load(checkpoint_fpath,map_location=torch.device('cuda:0'))
    else:
      checkpoint = torch.load(checkpoint_fpath,map_location=torch.device('cpu'))
    # initialize state_dict from checkpoint to model
    model.load_state_dict(checkpoint['state_dict'])
    # return model, optimizer, epoch value, min validation loss 
    return model

if __name__ == "__main__":
    device = "cpu"
    img = torch.randint(0,255,(1,3,512,512)).float()
    model = load_ckp(model_path, model, device=device)
    model.eval()
    out = model(img)
    torch.onnx.export(
        model,
        img,
        "models/dresunet.onnx",
        verbose=True,
        input_names = ['in_image'],
        output_names = ['out_mask','out_prob']
    )
