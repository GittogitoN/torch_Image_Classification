# -*- coding: <encoding name> -*-
# ライブラリのインポート
import torch.nn as nn
from torchvision import datasets, models, transforms
import torch

# 事前学習無のResNet18作成,転移学習あり
#model_ft = models.resnet18(pretrained=True)

#model_ft = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
#print(model_ft)




# 定数設定
DEVICE = "cpu"
TARGET_NUM = 4

# モデル作成関数の定義
def get_model(target_num):
    #for get model
    
    #model_ft = models.resnet18(pretrained=isPretrained)
    model_ft = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model_ft.fc = nn.Linear(512, target_num)
    model_ft = model_ft.to(DEVICE)
    return model_ft


model = get_model(TARGET_NUM)



checkpoint=torch.load('original_model_29.pth')

model.load_state_dict(checkpoint)




from PIL import Image



# 2. 学習時と同様の前処理を適用
data_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


file_list = []
pred_list = []

import glob
import numpy as np
import pandas as pd

# 3. 入力画像の読み込みと前処理
test_data_dir='.\02_PBL02_data\test'
for file in glob.glob(test_data_dir + '/*'):
    image_data = file
    filename = file.split('/')[-1]
    print(filename)
    image = Image.open(filename)
    image = data_transform(image)  # 前処理を適用

    # 4. バッチの次元を追加（モデルはバッチを受け入れるため）
    image = image.unsqueeze(0)  # バッチサイズを1に設定


    # 5. GPUを使用する場合、デバイスを設定
    device = torch.device("cpu")
    model.to(device)
    image = image.to(device)

    # 6. 推論を実行
    model.eval()  # モデルを推論モードに切り替え
    with torch.no_grad():  # 勾配計算を無効化
        output = model(image)

    # 7. 出力から予測結果を取得
    _, predicted_class = torch.max(output, 1)
    print("予測クラスインデックス:", predicted_class.item())

    # 8. クラスインデックスをクラス名にマッピングする場合
    class_names = ['00_bridge', '01_horn', '02_potato','03_regular' ]  # クラス名のリスト
    predicted_class_name = class_names[predicted_class.item()]
    print("予測クラス名:", predicted_class_name)

    # *bridge, horn, potatoを不良（'1'）に、regularを良品（'0'）に変換。
    if predicted_class_name=="00_bridge":
        judge=1
    elif predicted_class_name=='01_horn':
        judge=1
    elif predicted_class_name=='02_potato':
        judge=1
    else:
        judge=0

    pred_list.append(judge)
    file_list.append(file)
    #"]},{"cell_type":"codeexecution_count":null,"metadata":{"id":"d_1242zKUbtn"},"outputs":[],"source":["#判別結果をDataFrameに変換し、tsvファイルに出力
df = pd.DataFrame([file_list, pred_list]).T
df.to_csv('my_submission.tsv',
         index=False,
         header=False,
         sep='\t')















    
