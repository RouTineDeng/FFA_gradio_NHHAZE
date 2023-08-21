import os
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as tfs
import matplotlib.pyplot as plt
import gradio as gr
import io
from net.models import *  # 请根据您的模型导入正确的模块
from flask import request
# 加载模型和相关配置
abs = os.getcwd() + '/'
gps = 3
blocks = 20
dataset = 'NH'  # 请根据您的数据集配置
model_dir = abs + f'trained_models/{dataset}_train_ffa_{gps}_{blocks}.pk'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
ckp = torch.load(model_dir, map_location=device)
net = FFA(gps=gps, blocks=blocks)
net = nn.DataParallel(net)
net.to(device)
net.load_state_dict(ckp['model'])
net.eval()

def predict_haze(img):
    img = tfs.Compose([
    tfs.ToTensor(),
    tfs.Normalize(mean=[0.64, 0.6, 0.58],std=[0.14,0.15, 0.152])
    ])(img)[None,::]
    with torch.no_grad():
        pred = net(img)
    ts = torch.squeeze(pred.clamp(0, 1).cpu())
    # 将PyTorch Tensor转换为PIL Image
    pil_image = tfs.ToPILImage()(ts)

    # 返回预测结果
    return pil_image

# 创建Gradio界面
iface = gr.Interface(
    fn=predict_haze,
    inputs=gr.inputs.Image(type='pil'),  # 指定输入类型为PIL Image
    outputs=gr.outputs.Image(type="pil"),  # 指定输出类型为PIL Image
    live=True,
    capture_session=True,
)

# 启动Gradio应用程序
#iface.launch()
gr.close_all()  # 关闭所有正在运行的端口
iface.launch(share=True)