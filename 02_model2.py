# ライブラリのインポート
import torch.nn as nn
from torchvision import datasets, models, transforms
import torch

# 事前学習無のResNet18作成,転移学習あり
#model_ft = models.resnet18(pretrained=True)

#model_ft = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
#print(model_ft)



#データセットの作成
# transforms定義
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(256),#画像のピクセルをリサイズ
        transforms.RandomHorizontalFlip(p=0.5),#50%を反転して汎化性能up
        transforms.ToTensor(),#テンソル化
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),#、画像の輝度値を平均0、分散1に調整
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
}


#データセットの作成
# ライブラリのインポート
import pandas as pd
import os
#import zipfile
from torch.utils.data import Dataset
from PIL import Image


# dataset作成
image_datasets = {
    'train': datasets.ImageFolder('./image_Mizumashi/train',data_transforms['train']),
    'val': datasets.ImageFolder('./image_Mizumashi/val',data_transforms['val'])
}
#print(image_datasets)
#print(image_datasets['train'].class_to_idx)
#print(type(image_datasets))
#qprint(image_datasets['train'].samples[0:5])

#exit()

# dataloaders作成
image_dataloaders = {
    'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size=5,shuffle=True, num_workers=0, drop_last=True),
    'val': torch.utils.data.DataLoader(image_datasets['val'], batch_size=5,shuffle=False, num_workers=0, drop_last=True),
}

"""
for i ,(inputs,labels) in enumerate(image_dataloaders['train']):
            print(inputs)
            print(labels)
            if i == 0:
                break
"""
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


# 最適化関数定義
import torch.optim as optim
optimizer = optim.SGD(model.parameters(),lr=0.001, momentum=0.9)#モーメンタムは0.9~0.99がいいらしい

# loss関数定義
criterion = nn.CrossEntropyLoss()
dataset_sizes = {'train': len(image_datasets['train']), 'val': len(image_datasets['val'])}

print(image_datasets['val'].samples)
#print(len(image_datasets['val'].samples))
#print(len(image_datasets['train'].samples))#220:55=220/5
#exit()


#間違えリスト
misclassified_images = {'train': [], 'val': []}



#精度算出用関数
from sklearn.metrics import precision_score, recall_score, f1_score

def calculate_metrics(true_labels, predicted_labels):
    precision = precision_score(true_labels, predicted_labels, average='weighted')#適合率(TP/(TP+FP)
    recall = recall_score(true_labels, predicted_labels, average='weighted')#再現率(TP/(TP+FN)
    f1 = f1_score(true_labels, predicted_labels, average='weighted')#2*precision*recall/(precision+recall)
    return precision, recall, f1



# モデル訓練用関数
def train_model(model, criterion, optimizer, num_epochs=5,is_saved = False):
    best_acc = 0.0

    # エポック数だけ下記工程の繰り返し
    for epoch in range(num_epochs):

        for phase in ['train', 'val']:
            print('{}:フェイズ'.format(phase))

            # 訓練フェイズと検証フェイズの切り替え
            if phase == 'train':
                model.train() 
            else:
                model.eval()  

            running_loss = 0.0
            running_corrects = 0
            true_labels=[]#正解ラベルリスト
            predicted_labels=[]#推測らべるリス

            # dataloadersからバッチサイズだけデータ取り出し、下記工程（1−5）の繰り返し
            for i,(inputs, labels) in enumerate(image_dataloaders[phase]):#これがデータローダ
                #print(inputs,labels)
                
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)

                # 1. optimizerの勾配初期化
                optimizer.zero_grad()
                    
                # 2.モデルに入力データをinputし、outputを取り出す
                outputs = model(inputs)
                #print(outputs)#バッチサイズの数出てくる
                _, preds = torch.max(outputs, 1)
                #print(preds)

                # 3. outputと正解ラベルから、lossを算出
                loss = criterion(outputs, labels)

                #print("out",outputs)0
                #print("preds",preds)
                #print("label",labels)
                #print("inp",inputs)

                """
                print(image_datasets['val'].samples[i*4-3])
                print(image_datasets['val'].samples[i*4-2])
                print(image_datasets['val'].samples[i*4-1])
                print(image_datasets['val'].samples[i*4])
                """


                
                print('   loaders:{}回目'.format(i+1)  ,'   loss:{}'.format(loss))
                #exit()


                #print(preds[0])
                #print(labels.data[0])

                if phase == 'train':                        
                    # 4. 誤差逆伝播法により勾配の算出
                    loss.backward()
                    # 5. optimizerのパラメータ更新
                    optimizer.step()


                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                true_labels.extend(labels.cpu().numpy())#正解ラベルを追加
                predicted_labels.extend(preds.cpu().numpy())#推測ラベルを追加

                

            print(dataset_sizes[phase])
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]#一致したもの/全数
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            #exit()

            # C. 今までのエポックでの精度よりも高い場合はモデルの保存
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                if(is_saved):
                    torch.save(model.state_dict(), './original_model_{}.pth'.format(epoch))

            precision, recall, f1 = calculate_metrics(true_labels, predicted_labels)
            print('Precision: {:.4f}, Recall: {:.4f}, F1-Score: {:.4f}'.format(precision, recall, f1))


    print('Best val Acc: {:4f}'.format(best_acc))


train_model(model, criterion, optimizer, num_epochs=30,is_saved = True)


"""
# エポックごとに誤分類画像のリストを表示
for phase in ['train', 'val']:
    print(f'{phase} misclassified images:')
    for image_path in misclassified_images[phase]:
        print(image_path)

"""














    
