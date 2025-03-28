from pathlib import Path
from torch.utils.tensorboard import SummaryWriter

class Logger:
    def __init__(self, model, log_dir=Path("output/logs")):
        self.model = model
        self.log_dir = log_dir
        self.writer = None
    
    def write_train(self, epoch, total_epoch, train_loss, val_loss=None, val_acc=None):
        if self.writer is None:
            self.writer = SummaryWriter(log_dir=self.log_dir)
        if val_loss is not None and val_acc is not None:
            self.writer.add_scalars("Loss", {
                "Train": train_loss,
                "Val": val_loss
            }, epoch)
            self.writer.add_scalar("Accuracy/Val", val_acc, epoch)
            print(f"Epoch {epoch}/{total_epoch}, Train Loss: {train_loss}, Val Loss: {val_loss}, Val Acc: {val_acc}")
        else:
            self.writer.add_scalar("Loss/Train", train_loss, epoch)
            print(f"Epoch {epoch}/{total_epoch}, Train Loss: {train_loss}")
        
    def write_test(self, epoch, total_epoch, test_loss, test_acc):
        if self.writer is None:
            self.writer = SummaryWriter(log_dir=self.log_dir)
        self.writer.add_scalar("Loss/Test", test_loss, epoch)
        self.writer.add_scalar("Accuracy/Test", test_acc, epoch)
        print(f"Epoch {epoch}/{total_epoch}, Test Loss: {test_loss}, Test Acc: {test_acc}")
        
    def close(self):
        self.writer.close()