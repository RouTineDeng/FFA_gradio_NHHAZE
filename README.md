# 使用 Gradio进行FFA 模型测试

这个项目演示了如何使用 Gradio 来测试 FFA-Net（Feature Fusion Attention Network）模型的效果。FFA-Net是一个用于图像去雾的深度学习模型，您可以使用 Gradio 轻松上传图像并查看模型的预测效果。

## 快速开始

### 环境设置

首先，确保您的环境满足以下要求：

- Python 3.10
- 安装 Gradio 和其他依赖项：

```bash
pip install gradio
# 还需要安装其他所需的库，根据您的项目需求
```

### 运行示例

1. 克隆此仓库：

   ```
   https://github.com/RouTineDeng/FFA_gradio_NHHAZE.git
   cd FFA_gradio_NHHAZE
   ```

2. 运行 Gradio 应用程序

   ```
   gradio demo.py
   ```

3. 打开浏览器并访问 `http://localhost:7860`，您将能够上传图像并查看 FFA 模型的预测效果。

**为了方便查看测试效果，此处给出demo允许网址供给查看效果，测试图像已放入image文件夹中**

## 雾气场景迁移

目前默认使用FFA模型训练得到的NH-HAZE代码文件对图像进行去雾，它更倾向于非均匀薄雾图像去雾。如果您想要使用自己训练的模型，可以在 `demo.py` 中将模型替换为您自己的模型。确保您的模型接受并返回与示例代码相同的图像格式。

```python
# 加载模型和相关配置
abs = os.getcwd() + '/'
gps = 3
blocks = 19
dataset = 'its'  # 请根据您的数据集配置
model_dir = abs + f'trained_models/{dataset}_train_ffa_{gps}_{blocks}.pk'
```

