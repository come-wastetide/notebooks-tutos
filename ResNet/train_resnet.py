import pytorch_lightning as pl
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
import torchvision
import torchvision.models as models
import torchmetrics
import torch.nn.functional as F
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from PIL import ImageFile
from torch import nn, utils


import torch


size = (620,620)


def make_train_valid_test_data(DATASET_PATH,size=size):
    

    '''
    We assume that the data is sorted inside the data_path file according to this architecture :

    data
    -Verre
    --Verre01
    --Verre02
    ...
    -Mégots
    --Mégots01
    --Mégots02
    ...
    ...
    '''

    

    data_transforms = {
        'train': transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomResizedCrop(size, antialias=None),
            transforms.RandomHorizontalFlip(),
            # transforms.Grayscale(),
            # transforms.Normalize(([0.6731, 0.6398, 0.6048]), ([0.1944, 0.1931, 0.2049]))
        ]),
        'test': transforms.Compose([
            transforms.Resize(size, antialias = None),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
    }

    train_dataset = torchvision.datasets.ImageFolder(DATASET_PATH, transform=data_transforms['train'])
    test_dataset = torchvision.datasets.ImageFolder(DATASET_PATH, transform=data_transforms['test'])

    torch.manual_seed(1)
    np.random.seed(1)
    indices = np.random.permutation(len(train_dataset)).tolist()
    LABELS = train_dataset.classes


    test_ratio = 0.2
    test_border = len(train_dataset) - int(len(train_dataset) * (test_ratio))

    train_data = torch.utils.data.Subset(train_dataset, indices[:test_border])
    test_data = torch.utils.data.Subset(test_dataset, indices[test_border:])

    train_size = int(0.9 * len(train_data))
    val_size = len(train_data) - train_size

    train_data, val_data = utils.data.random_split(train_data, [train_size, val_size])

    print(f'there is {len(train_data)} images for training')
    print(f'there is {len(test_data)} images for testing')

    return train_data,val_data,test_data


def make_loader(data,batch_size, shuffle=False):

        if shuffle:
            return torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True, num_workers=3)

        return torch.utils.data.DataLoader(data, batch_size=64, num_workers=3)



from pytorch_lightning.callbacks import Callback

class MetricMonitor(Callback):
    def __init__(self):
        self.history = []
        self.epoch = 0

    def on_train_epoch_end(self, trainer, pl_module):
        elogs = {item: float(value) for (item, value) in trainer.logged_metrics.items()}
        print(f"Epoch [{self.epoch}] train_loss: {elogs['train_loss_epoch']:.3f}, val_loss: {elogs['val_loss']:.3f}, train_acc: {elogs['train_acc']:.3f}, val_acc: {elogs['val_acc']:.3f}")
        self.epoch += 1
        self.history.append(elogs)




class LigResNet(pl.LightningModule):
    def __init__(self, lr, num_class, *args, **kwargs):
        super().__init__()
        
        self.save_hyperparameters()
        
        self.model = models.resnet50(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_class)
        
        self.train_acc = torchmetrics.Accuracy(task='multiclass', num_classes=10)
        self.val_acc = torchmetrics.Accuracy(task='multiclass', num_classes=10)
        self.test_acc = torchmetrics.Accuracy(task='multiclass', num_classes=10)
    
    def forward(self, x):
        return self.model(x)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5)
        return [optimizer], [scheduler]
    
    def training_step(self, batch, batch_idx):
        X, y = batch
        logits = self.model(X)
        loss = F.cross_entropy(logits, y)
        
        self.train_acc(torch.argmax(logits, dim=1), y)
        
        self.log('train_loss', loss.item(), on_epoch=True)
        self.log('train_acc', self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        X, y = batch
        logits = self.model(X)
        loss = F.cross_entropy(logits, y)
        
        self.val_acc(torch.argmax(logits, dim=1), y)
        
        self.log('val_loss', loss.item(), on_epoch=True)
        self.log('val_acc', self.val_acc, on_epoch=True)
    
    def test_step(self, batch, batch_idx):
        X, y = batch
        logits = self.model(X)
        loss = F.cross_entropy(logits, y)
        
        self.test_acc(torch.argmax(logits, dim=1), y)
        
        self.log('test_loss', loss.item(), on_epoch=True)
        self.log('test_acc', self.test_acc, on_epoch=True)
    
    def predict_step(self, batch, batch_idx):
        X, y = batch
        preds = self.model(X)
        return preds

class LigResNeXt(pl.LightningModule):
    def __init__(self, lr, num_class, *args, **kwargs):
        super().__init__()
        
        self.save_hyperparameters()
        
        self.model = models.resnext50_32x4d(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_class)
        
        self.train_acc = torchmetrics.Accuracy(task='multiclass', num_classes=10)
        self.val_acc = torchmetrics.Accuracy(task='multiclass', num_classes=10)
        self.test_acc = torchmetrics.Accuracy(task='multiclass',num_classes=10)
    
    def forward(self, x):
        return self.model(x)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5)
        return [optimizer], [scheduler]
    
    def training_step(self, batch, batch_idx):
        X, y = batch
        logits = self.model(X)
        loss = F.cross_entropy(logits, y)
        
        self.train_acc(torch.argmax(logits, dim=1), y)
        
        self.log('train_loss', loss.item(), on_epoch=True)
        self.log('train_acc', self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        X, y = batch
        logits = self.model(X)
        loss = F.cross_entropy(logits, y)
        
        self.val_acc(torch.argmax(logits, dim=1), y)
        
        self.log('val_loss', loss.item(), on_epoch=True)
        self.log('val_acc', self.val_acc, on_epoch=True)
    
    def test_step(self, batch, batch_idx):
        X, y = batch
        logits = self.model(X)
        loss = F.cross_entropy(logits, y)
        
        self.test_acc(torch.argmax(logits, dim=1), y)
        
        self.log('test_loss', loss.item(), on_epoch=True)
        self.log('test_acc', self.test_acc, on_epoch=True)
    
    def predict_step(self, batch, batch_idx):
        X, y = batch
        preds = self.model(X)
        return preds

    #num_class = len(LABELS)

def create_model(num_class,lr=0.00005,model_type = 'Resnet'):

    if model_type=='Resnet':
        model_1 = LigResNet(lr=lr, num_class=num_class)
        model_1.model.fc
        return model_1

    elif model_type=='Resnext':
        model_2 = LigResNeXt(lr=lr, num_class=num_class)
        model_2.model.fc
        return model_2



'''from pytorch_lightning.loggers import NeptuneLogger

neptune_logger = NeptuneLogger(
    project="wastetide/classifier",
    api_key="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI3ZDk0YjQxNS1lY2MzLTRlOTUtYTk3ZC1jYzNkYjQxMmNhMTQifQ==",
    log_model_checkpoints=False,
)'''




def train(train_loader,val_loader,model,max_epochs=25, limit_train_batches=100,default_root_dir='./logs/resnet'):

    mm = MetricMonitor()
    csv_log = CSVLogger('logs', name='metric')
    es = EarlyStopping('val_loss', patience=3)
    mc = ModelCheckpoint(filename='{epoch}-{val_loss}', monitor='val_loss', save_top_k=3)


    trainer = pl.Trainer(
        #logger=neptune_logger,
        accelerator='gpu',
        limit_train_batches=100,
        max_epochs=max_epochs,
        devices=1,
        callbacks=[mm, es, mc],
        default_root_dir='./logs/resnet'
    )
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments=True
    ImageFile.LOAD_TRUNCATED_IMAGES = True

    torch.cuda.empty_cache()

    trainer.fit(model, train_loader, val_loader)

    return trainer





    


