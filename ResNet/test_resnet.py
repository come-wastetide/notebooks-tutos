import sklearn.metrics as metrics
import torch
import numpy as np
import itertools
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import pytorch_lightning as pl
from train_resnet import LigResNeXt





def test(trainer,model,test_loader,ckpt_path='best'):

    trainer.test(model, test_loader, ckpt_path=ckpt_path)
    

class LigResNeXt(pl.LightningModule):
    def __init__(self, lr, num_class, *args, **kwargs):
        super().__init__()
        
        self.save_hyperparameters()
        
        self.model = models.resnext50_32x4d(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_class)
        
        self.train_acc = torchmetrics.Accuracy(task='multiclass', num_classes=7)
        self.val_acc = torchmetrics.Accuracy(task='multiclass', num_classes=7)
        self.test_acc = torchmetrics.Accuracy(task='multiclass',num_classes=7)
    
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



def print_confusion_matrix(labels,predictions,class_names):
    
    

    ''' 
    To complete ! 
    

    # Charger le modèle entraîné
    model = LitResNeXt.load_from_checkpoint(path_to_checkpoints)
    
    # Désactiver le mode d'entraînement
    model.eval()


    # Prédire les étiquettes pour les données de test
    predictions = []
    labels = []
    with torch.no_grad():
        for batch in test_loader:
            images, targets = batch
            outputs = model(images)
            _, predicted = torch.max(outputs, dim=1)
            predictions.extend(predicted.cpu().numpy())
            labels.extend(targets.cpu().numpy())'''

    # Calculer la matrice de confusion
    confusion_matrix = metrics.confusion_matrix(labels, predictions)

    
    # Define the confusion matrix
    cm = np.array(confusion_matrix)

    # Normalize the confusion matrix to percentages
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

    # Plot the confusion matrix
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm_normalized, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    # Set the labels for the rows and columns
    ax.set(xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel='True label',
        xlabel='Predicted label')

    # Set the threshold for different colors
    threshold = cm_normalized.max() / 2.

    # Add the values to the cells
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, f'{cm[i, j]} ({cm_normalized[i, j]:.1f}%)',
                horizontalalignment="center",
                color="white" if cm_normalized[i, j] > threshold else "black")

    # Add a title
    plt.title('Confusion Matrix')

    # Show the plot
    plt.show()

    # Afficher la matrice de confusion
    print(confusion_matrix)