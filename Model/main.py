import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

# 공개 데이터셋에서 학습 데이터 내려 받기
training_data = datasets.FashionMNIST(
    root="data",
    train = True,
    download=True,
    transform=ToTensor(),
)
print(training_data)

# 공개 데이터셋에서 테스트 데이터 내려 받기
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)

batch_size = 64

# 데이터로더를 생성
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

for X,y in test_dataloader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break

# 학습에 사용할 CPU나 GPU, MPS 장치 얻기 => torch.cuda 또는 torch.backends.mps 가 사용 가능한지 확인 else : Cpu 사용
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

# 모델 정의 => nn.Module의 하위클래스로 정의
class NeuralNetwork(nn.Module):

    # __init__에서 신경망 계층들을 초기화
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512,512),
            nn.ReLU(),
            nn.Linear(512,10)
        )

    # nn.Model을 상속받은 모든 클래스를 forward에 입력 데이터에 대한 연산들을 구현
    def forward(self,x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to(device)
print(model)

# 모델 학습 => 손실함수(loss function)와 옵티마이저(optimizer)가 필요
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(),lr=1e-3)

# 각 학습 단계에서 모델은 학습 데이터셋에 대한 예측을 수행
# => 예측 오류를 역전파하여 모델의 매개변수를 조정
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X,y) in enumerate(dataloader):
        X,y = X.to(device), y.to(device)

        # 에측 오류 계산
        pred = model(X)
        loss = loss_fn(pred,y)

        # 역전파
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")

# 모델이 학습하고 있는지 확인 => 테스트 데이터셋으로 모델의 성능 확인
def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0,0
    with torch.no_grad():
        for X, y in dataloader:
            X, y  = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

# 학습 단계는 여러번 반복 => 에폭에서 모델은 더 나은 예측을 하기 위해 매개변수를 학습
# => 각 에폭마다 모델의 정확도(accuracy)와 손실(loss)을 출력
epochs = 5
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
print("Done!")

# 모델 저장하는 일반적인 방법 => 내부 상태 사전을 직렬화
torch.save(model.state_dict(),"model.pth")
print("Saved PyTorch Model State to model.pth")

# 모델 불러오기 => 모델 구조를 다시 만들고 상태 사전을 모델에 불러오는 과정에 포함됨
model = NeuralNetwork().to(device)
model.load_state_dict(torch.load("model.pth"))

# 이제 모델을 사용해서 예측 할 수 있다.

classes = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

model.eval()
x, y= test_data[0][0], test_data[0][1]
with torch.no_grad():
    x = x.to(device)
    pred = model(x)
    predicted, actual = classes[pred[0].argmax(0)],classes[y]
    print(f'Predicted: "{predicted}", Actual : "{actual}"')