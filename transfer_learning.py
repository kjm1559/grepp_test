import torch
from torch import nn
import torch.nn.functional as F
from torchsummary import summary
from model import ConvNet
from utils import CustomDataset
from torchvision import datasets, transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from torch.ao.quantization import quantize_dynamic
from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantType, QuantFormat, CalibrationMethod
import onnxruntime as ort
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA
import seaborn as sns
import pandas as pd
import random

random.seed(0)
torch.manual_seed(0)
np.random.seed(0)

IMAGE_SIZE = 64
BATCH_SIZE = 16
NUM_WORKERS = 4 
EPOCH = 2
LEARING_RATE = 1e-3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def pca_draw(model, dataloader, title, path):
    """
    This function is ploting distribution of feature.
    
    Parameters:
        model : Pytorch model
        dataloader : Target data loader
        path : Save path
    """
    feature_maps = []
    
    def forward_hook(module, input, output):
        feature_maps.append(output)
        
    target_layer = model.conv3
    forward_hook = target_layer.register_forward_hook(forward_hook)
    forward_result = []
    labels = []
    images = []
    counter = 0
    with torch.no_grad():
        for X, y in dataloader:
            y = y.type(torch.LongTensor)
            X = X.to(device)
            y = y.to(device)
            y_pred = model(X)
            counter += X.shape[0]
            forward_result += feature_maps[-1]
            labels.append(y)
            images.append(X)
    images = torch.concat(images, dim=0)
    
    forward_result = torch.concat(feature_maps, dim=0).detach().cpu().mean(dim=[2, 3]).numpy()
    labels = torch.concat(labels).detach().cpu().numpy()
    
    # Calculate PCA
    pca = PCA(n_components=2)
    projected_data = pca.fit_transform(forward_result)
    label = ['apple', 'cherry', 'tomato']
    
    data_df = pd.DataFrame(projected_data, columns=['dim1', 'dim2'])
    data_df['class'] = [label[i] for i in labels]
    
    fig = sns.jointplot(x=data_df["dim1"], y=data_df["dim2"], hue=data_df["class"], kind="kde")
    fig.fig.suptitle(title)
    
    plt.tight_layout()
    plt.savefig(path)
    plt.cla()
    plt.clf() 
    forward_hook.remove()

def draw_gradCAM(model, transform, path, label):
    """ 
    This function is make gradCAM image.
    
    Parameters:
        model : Pytorch model
        transform : Transform method for input image
        path : image file path and name
    """
    target_data_list = ['data/train/apple/327_100.jpg', 'data/train/cherry/327_100.jpg', 'data/train/tomato/9_100.jpg']
    image_data_list = []
    original_image_list = []
    for t in target_data_list:
        with Image.open(t) as image:
            original_image_list.append(image.copy())
            image = transform(image)
            image_data_list.append(image)
    input_data = torch.stack(image_data_list)
    
    
    feature_maps = []
    gradients = []
    
    def forward_hook(module, input, output):
        feature_maps.append(output)

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])
    
    for param in model.conv3.parameters():
        param.requires_grad = True
    
    target_layer = model.conv3
    forward_hook = target_layer.register_forward_hook(forward_hook)
    backward_hook = target_layer.register_full_backward_hook(backward_hook)
    
    output = model(input_data.to(device))
    preds = output.argmax(dim=1)
    
    for i in range(len(original_image_list)):
        model.zero_grad()
        output[i, preds[i]].backward(retain_graph=True)

    # 4. Grad-CAM 계산 per image
    acts = feature_maps[0]     # (B, C, H, W)
    grads = gradients[0]       # (B, C, H, W)

    cams = []
    for i in range(len(original_image_list)):
        weights = gradients[i][i].mean(dim=(1, 2), keepdim=True)   # (C, 1, 1)
        cam = (weights * acts[i]).sum(dim=0)                # (H, W)
        cam = F.relu(cam)
        cam = cam / cam.max()
        cams.append(cam.detach().cpu().numpy())

    # 5. 시각화
    plt.figure(figsize=(8, 3))
    for i in range(len(original_image_list)):
        img_np = np.array(original_image_list[i].resize((IMAGE_SIZE, IMAGE_SIZE)))# / 255.0
        cam_resized = np.array(Image.fromarray(cams[i]).resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR))

        plt.subplot(1, len(original_image_list), i+1)
        plt.imshow(img_np)
        plt.imshow(cam_resized, cmap='jet', alpha=0.5)
        plt.title(f'Grad-CAM #{i} {label[preds[i]]}')
        plt.axis('off')

    plt.tight_layout()
    
    plt.savefig(path)
    plt.cla()
    plt.clf()
    for param in model.conv3.parameters():
        param.requires_grad = False
       
    forward_hook.remove()
    backward_hook.remove()

def test(model, dataloader, epoch=-1):
    """
    This function is for calculating validation accuracy.
    
    Parameters:
        model : pytorch model
        dataloader : dataloader for validation data set
        epoch : Current epoch
        
    Returns:
        accuracy : The average accuracy
    
    """
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    features = []
    labels = []
    model.fc.eval()
    acc = []
    with torch.no_grad():
        for X, y in dataloader:
            y = y.type(torch.LongTensor)
            X = X.to(device)
            y = y.to(device)
            y_pred = model(X)#.reshape((batch_size, -1))
            acc.append(y[torch.argmax(y_pred, dim=-1) == y].shape[0]/y.shape[0])
        print(f'eval acc: {torch.mean(torch.Tensor(acc)):>4}')
    return torch.mean(torch.Tensor(acc))

def onnx_test(path, dataloader):
    """
    This function is for testing quantized onnx model.
    
    Parameters:
        path : The path of onnx file.
        dataloader : The dataloader for testing.
        
    Returns:
        accuracy : The average accuracy
    """
    session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    acc = []
    y_pred_data = []
    y_data = []
    for X, y in dataloader:
        y_pred = session.run(None, {input_name: X.numpy()})
        acc.append(y[np.argmax(y_pred, axis=-1)[0] == y.numpy()].shape[0]/y.shape[0])
        y_pred_data += np.argmax(y_pred, axis=-1)[0].tolist()
        y_data += y.numpy().tolist()
    print(classification_report(np.stack(y_data), np.stack(y_pred_data)))
    return np.mean(acc)
        

def train(model, dataloader, epoch, loss_f, optimizer, test_dataloader=None):
    """
    This function is for training model.
    
    Parameters:
        model : The pytorch model
        dataloader : The dataset for training
        epoch : Current epoch
        loss_f : The loss function for training
        optimizer : the optimizer for training
        test_dataloader : For the testing model
    """
    losses = []
    size = len(dataloader.dataset)
    result_acc = []

    p_loss = 0
    for i in range(epoch):
        if test_dataloader is None:
            result_acc.append(test(model, dataloader, i))
        else:
            result_acc.append(test(model, test_dataloader, i))
        model.fc.train()  
        for batch, (x, y) in enumerate(dataloader):
            model.fc.zero_grad()
            y = y.type(torch.LongTensor)
            x = x.to(device)
            y = y.to(device)

            y_pred = model(x)

            loss = loss_f(y_pred, y)                    
            with torch.autograd.set_detect_anomaly(True):
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(loss.item())

            if batch % 5 == 0:
                # print(torch.argmax(y_pred, dim=-1), y)
                loss, current = loss.item(), batch * len(x)
                print(f"{i+1} epoch loss: {loss:>7f}, acc: {y[torch.argmax(y_pred, dim=-1) == y].shape[0]/y.shape[0]:>4} [{current:->5d}/{size:>5d}]")
                
class MyCalibrationDataReader(CalibrationDataReader):
    """The datareader for the testing onnx model"""
    def __init__(self, dataloader, input_name):
        super().__init__()
        self.dataloader = dataloader
        self.input_name = input_name
        self.iterator = iter(self.dataloader)

    def get_next(self):
        try:
            batch = next(self.iterator)
        except StopIteration:
            return None
        return {self.input_name: batch[0].cpu().numpy()}

    def __len__(self):
        return len(self.dataloader)

if __name__ == "__main__":
    # Define tranform method
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225], inplace=False),
    ])
    
    train_data = CustomDataset(True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True,
                                          num_workers=NUM_WORKERS, pin_memory=True)
    
    val_data = CustomDataset(False, transform=transform)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False,
                                          num_workers=NUM_WORKERS, pin_memory=True)
    
    model = ConvNet()
    model.load_state_dict(torch.load("conv_net.pth", map_location=device, weights_only=True))
    # Freez weights
    for param in model.parameters():
        param.requires_grad = False
    
    pca_draw(model.to(device), train_loader, 'Current model feature distribution', 'current_feature_scatter.png')
    
    draw_gradCAM(model, transform, 'before_train.png', ['apple', 'lychee', 'banana', 'cherry', 'orange'])
    
    # New head for 3 class
    model.fc = nn.Linear(64 * 8 * 8, 3)
    
    model = model.to(device)
    summary(model, input_size=(3, 64, 64))
    
    loss_f = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=LEARING_RATE, weight_decay=0.001)
    
    # training
    for i in range(EPOCH):
        train(model, train_loader, i, loss_f, optimizer)
        test(model, val_loader)
        
    draw_gradCAM(model, transform, 'after_train.png', ['apple', 'cherry', 'tomato'])
    
    # quantization
    device = 'cpu'
    # pytorch quantization
    # model_quantized = quantize_dynamic(model.to(device), {torch.nn.Linear}, dtype=torch.qint8)
    # print('============== quantization model test ==================')
    # test(model_quantized, val_loader)
    
    # ONNX export
    dummy_input = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)

    torch.onnx.export(
        model.to(device),           # 모델
        dummy_input,                # 입력값
        "transfer_fp32.onnx",       # 저장 파일명
        export_params=True,         # 파라미터 포함
        opset_version=13,           # ONNX opset version
        do_constant_folding=True,   # 상수 폴딩 최적화
        input_names=['input'],      # 입력 이름
        output_names=['output'],    # 출력 이름
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    
    # Quantization to Int8
    reader = MyCalibrationDataReader(val_loader, 'input')
    quantize_static(
        model_input="transfer_fp32.onnx",
        model_output="transfer_int8.onnx",
        calibration_data_reader=reader,
        quant_format=QuantFormat.QOperator,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        per_channel=False,
        reduce_range=False
    )
    
    # test quantization model
    print('============== quantization onnx model test ==================')
    print(onnx_test('transfer_int8.onnx', val_loader))
        
    