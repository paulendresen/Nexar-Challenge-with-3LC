import torch
from torch.amp import autocast, GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts
from typing import Dict, Any
from tqdm import tqdm
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import sklearn.metrics  # Add at the top with other imports
from scipy.special import expit

def event_loss(event_logits, event_occurs, mean=True):
    eps = 1e-8
    
    # Stage 1: Binary classification loss with logits
    classification_loss = F.binary_cross_entropy_with_logits(
        event_logits, 
        event_occurs.float(),
        reduction='none'
    )
    
      # Combine losses
    total_loss = classification_loss
    
    if mean:
        return total_loss.mean()
    else:
        return total_loss

class TrainingScheduler:
    def __init__(
        self,
        model: torch.nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        config: Dict[str, Any],
        test_mode: bool = False  # Add test mode flag
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Setup optimizer and scheduler
        self.optimizer = AdamW(
            model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
        
        # Setup scheduler based on config
        if config['scheduler_type'] == 'cosine':
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=config['T_max'],  # Number of epochs
                eta_min=config['min_lr']  # Minimum learning rate
            )
        else:  # 'cosine_warm_restarts'
            self.scheduler = CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=config['T_0'],  # Initial restart interval
                T_mult=config['T_mult'],  # Factor to increase T_0 after each restart
                eta_min=config['min_lr']  # Minimum learning rate
            )
        
        # Setup mixed precision training
        self.scaler = GradScaler()
       
        # Initialize best metrics
        self.best_val_loss = float('inf')
        self.best_val_acc = 0
        self.best_val_f1 = 0
        self.best_aps_score = 0
        self.patience = config['patience']
        self.patience_counter = 0
        self.best_optimal_f1 = 0  # Add new tracking variable
        
        # Update criterion for continuous value prediction
        #self.criterion =  nn.BCEWithLogitsLoss() #FocalLoss()#CustomRegressionLoss(alert_weight=2.5, crash_weight=1.5)
        self.test_mode = test_mode
    
    def train(self, epochs: int):
        self.current_epoch = 0
        
        for epoch in range(epochs):
            self.current_epoch = epoch
            print(f"\nEpoch {epoch + 1}/{epochs}")
            
            # Training phase
            print("Training...")
            train_loss, train_acc, train_prec, train_rec, train_f1 = self._train_epoch()
            
            # Validation phase
            print("Validating...")
            val_loss, predictions, val_acc, val_prec, val_rec, val_f1, val_aps, optimal_threshold, class_metrics = self._validate()
            
            # Print metrics with separate accuracies
            self._print_metrics(
                epoch, 
                train_loss, 
                (train_acc, train_prec, train_rec, train_f1),
                val_loss, 
                (val_acc, val_prec, val_rec, val_f1, val_aps, class_metrics)
            )
            
            # Early stopping and checkpoint logic
            checkpoint_info = []
            is_best = False
            
            # Check for best metrics
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                checkpoint_info.append(f"loss_{val_loss:.2f}")
                is_best = True
                
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                checkpoint_info.append(f"acc_{val_acc*100:.2f}")
                is_best = True

            # Add F1 check
            if val_f1 > self.best_val_f1:
                self.best_val_f1 = val_f1
                checkpoint_info.append(f"f1_{val_f1*100:.2f}")
                is_best = True

            # Add APS check
            if val_aps > self.best_aps_score:
                self.best_aps_score = val_aps
                checkpoint_info.append(f"aps_{val_aps:.4f}")
                is_best = True
            
            if True: #is_best:
                self.patience_counter = 0
                metric_str = '_'.join(checkpoint_info)
                # Add class-specific metrics to filename
                class_metrics_str = (
                    f"_CP_{class_metrics['collision']['precision']*100:.2f}"
                    f"_CR_{class_metrics['collision']['recall']*100:.2f}"
                    f"_NCP_{class_metrics['no_collision']['precision']*100:.2f}"
                    f"_NCR_{class_metrics['no_collision']['recall']*100:.2f}"
                    f"_APS_{class_metrics['collision']['average_precision_score']*100:.2f}"
                    f"_F1_{class_metrics['collision']['f1']*100:.2f}"
                    f"_ACC_{class_metrics['accuracy']*100:.2f}"
                )
                self._save_checkpoint(epoch, metric_str + class_metrics_str)
                print(f"New best model saved! Best {metric_str}: "
                      f"(recall: {val_rec:.4f}, loss: {val_loss:.4f}, "
                      f"acc: {val_acc:.4f}, f1: {val_f1:.4f}, aps: {val_aps:.4f})")
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.patience:
                    print(f"Early stopping triggered after {epoch + 1} epochs")
                    break
            
            # Update scheduler
            self.scheduler.step()
    

    def _train_epoch(self):
        self.model.train()
        running_loss = 0.0
        
        # Reset gradients at the start of each epoch
        self.optimizer.zero_grad()
        
        # Tracking metrics
        correct_predictions = 0
        total_predictions = 0
        
        # For precision and recall calculation for both classes
        true_positives = 0
        true_negatives = 0
        false_positives = 0
        false_negatives = 0
        
        pbar = tqdm(enumerate(self.train_loader), total=len(self.train_loader), 
                   desc="Training", leave=False)
        
        scaler = torch.amp.GradScaler()
        for batch_idx, (videos, event_occurs) in pbar:
            # Unpack the tuple and move each tensor to device
            short_frames = videos
            short_frames = short_frames.to(self.device)
            #long_frames = long_frames.to(self.device)
            event_occurs = event_occurs.to(self.device)
            
            with autocast(device_type='cuda', enabled=True):
                event_logits = self.model(short_frames) # short_frames, long_frames))
                event_logits = event_logits.squeeze(1)
                loss = event_loss(event_logits, event_occurs)
            
            # Add to running loss before normalizing
            running_loss += loss.item()
            
            # Normalize loss by gradient accumulation steps for backward
            loss = loss / self.config['gradient_accumulation_steps']
            scaler.scale(loss).backward()
            
            # Only update weights after accumulating gradients
            if ((batch_idx + 1) % self.config['gradient_accumulation_steps'] == 0) or (batch_idx + 1 == len(self.train_loader)):
                scaler.unscale_(self.optimizer)
                scaler.step(self.optimizer)
                scaler.update()
                self.optimizer.zero_grad()

            # Flatten tensors for metrics
            event_prob = torch.sigmoid(event_logits).squeeze()  # Shape: (batch_size,)
            binary_preds = (event_prob > 0.5).float()  # Shape: (batch_size,)

            # Overall accuracy
            correct_predictions += (binary_preds == event_occurs).float().sum().item()
            total_predictions += event_occurs.size(0)
            
            # Calculate metrics for both classes
            true_positives += ((binary_preds == 1) & (event_occurs == 1)).float().sum().item()
            true_negatives += ((binary_preds == 0) & (event_occurs == 0)).float().sum().item()
            false_positives += ((binary_preds == 1) & (event_occurs == 0)).float().sum().item()
            false_negatives += ((binary_preds == 0) & (event_occurs == 1)).float().sum().item()
            
            # Calculate metrics for both classes
            collision_precision = true_positives / (true_positives + false_positives + 1e-8)
            collision_recall = true_positives / (true_positives + false_negatives + 1e-8)
            
            no_collision_precision = true_negatives / (true_negatives + false_negatives + 1e-8)
            no_collision_recall = true_negatives / (true_negatives + false_positives + 1e-8)
            
            # Calculate macro-averaged metrics
            macro_precision = (collision_precision + no_collision_precision) / 2
            macro_recall = (collision_recall + no_collision_recall) / 2
            final_f1 = 2 * (macro_precision * macro_recall) / (macro_precision + macro_recall + 1e-8)
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{correct_predictions / total_predictions * 100:.2f}%',
                'precision': f'{macro_precision*100:.2f}%',
                'recall': f'{macro_recall*100:.2f}%',
                'f1': f'{final_f1*100:.2f}%',
   
            })
                
        # Calculate final metrics
        accuracy = correct_predictions / total_predictions
        final_precision = macro_precision
        final_recall = macro_recall
        
        print(f"Learning rate: {self.optimizer.param_groups[0]['lr']:.8f}")
        return (running_loss / len(self.train_loader), accuracy, 
                final_precision, final_recall, final_f1)

    def _validate(self):
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_labels = []
        
        # Tracking metrics
        correct_predictions = 0
        total_samples = 0
        
        # For precision and recall calculation for both classes
        true_positives = 0
        true_negatives = 0
        false_positives = 0
        false_negatives = 0
        
        pbar = tqdm(self.val_loader, desc="Validating", leave=False)
        
        with torch.no_grad():
            for videos, event_occurs in pbar:
                # Unpack the tuple and move each tensor to device
                short_frames = videos
                short_frames = short_frames.to(self.device)
                # long_frames = long_frames.to(self.device)
                event_occurs = event_occurs.to(self.device)
                
                with autocast(device_type='cuda', enabled=True):
                    event_logits = self.model(short_frames)
                    event_logits = event_logits.squeeze(1)
                    loss = event_loss(event_logits, event_occurs)
                
                total_loss += loss.item()
                
                # Flatten tensors for metrics
                event_prob = torch.sigmoid(event_logits).squeeze()  # Shape: (batch_size,)
                binary_preds = (event_prob > 0.5).float()  # Shape: (batch_size,)
                
                # Overall accuracy
                correct_predictions += (binary_preds == event_occurs).float().sum().item()
                total_samples += event_occurs.size(0)
                
                # Calculate metrics for both classes
                true_positives += ((binary_preds == 1) & (event_occurs == 1)).float().sum().item()
                true_negatives += ((binary_preds == 0) & (event_occurs == 0)).float().sum().item()
                false_positives += ((binary_preds == 1) & (event_occurs == 0)).float().sum().item()
                false_negatives += ((binary_preds == 0) & (event_occurs == 1)).float().sum().item()
                
                # Store raw predictions and labels for metrics
                all_predictions.extend(event_prob.cpu().numpy())
                all_labels.extend(event_occurs.cpu().numpy())
                
                # Calculate metrics for both classes
                collision_precision = true_positives / (true_positives + false_positives + 1e-8)
                collision_recall = true_positives / (true_positives + false_negatives + 1e-8)
                
                no_collision_precision = true_negatives / (true_negatives + false_negatives + 1e-8)
                no_collision_recall = true_negatives / (true_negatives + false_positives + 1e-8)
                
                # Calculate macro-averaged metrics
                macro_precision = (collision_precision + no_collision_precision) / 2
                macro_recall = (collision_recall + no_collision_recall) / 2
                final_f1 = 2 * (macro_precision * macro_recall) / (macro_precision + macro_recall + 1e-8)
                
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{correct_predictions / total_samples * 100:.2f}%',
                    'precision': f'{macro_precision*100:.2f}%',
                    'recall': f'{macro_recall*100:.2f}%',
                    'f1': f'{final_f1*100:.2f}%',
       
                })
        
        # Calculate final metrics
        accuracy = correct_predictions / total_samples
        final_precision = macro_precision
        final_recall = macro_recall
        avg_loss = total_loss / len(self.val_loader)
        
        # First convert lists to numpy arrays
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        
        # Find optimal threshold by testing range of thresholds
        thresholds = np.arange(0, 1.01, 0.01)  # Test thresholds from 0 to 1
        best_f1 = 0
        optimal_threshold = 0.5
        
        for threshold in thresholds:
            binary_preds = (all_predictions > threshold).astype(float)
            
            # Calculate metrics for this threshold
            tp = np.sum((binary_preds == 1) & (all_labels == 1))
            tn = np.sum((binary_preds == 0) & (all_labels == 0))
            fp = np.sum((binary_preds == 1) & (all_labels == 0))
            fn = np.sum((binary_preds == 0) & (all_labels == 1))
            
            # Calculate precision and recall for both classes
            coll_prec = tp / (tp + fp + 1e-8)
            coll_rec = tp / (tp + fn + 1e-8)
            no_coll_prec = tn / (tn + fn + 1e-8)
            no_coll_rec = tn / (tn + fp + 1e-8)
            
            # Calculate macro F1
            macro_prec = (coll_prec + no_coll_prec) / 2
            macro_rec = (coll_rec + no_coll_rec) / 2
            f1 = 2 * (macro_prec * macro_rec) / (macro_prec + macro_rec + 1e-8)
            
            if f1 > best_f1:
                best_f1 = f1
                optimal_threshold = threshold
        
        # Calculate final metrics using optimal threshold
        final_preds = (all_predictions > optimal_threshold).astype(float)
        tp = np.sum((final_preds == 1) & (all_labels == 1))
        tn = np.sum((final_preds == 0) & (all_labels == 0))
        fp = np.sum((final_preds == 1) & (all_labels == 0))
        fn = np.sum((final_preds == 0) & (all_labels == 1))
        
        # Calculate accuracy with optimal threshold
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        
        # Calculate final metrics with optimal threshold
        collision_precision = tp / (tp + fp + 1e-8)
        collision_recall = tp / (tp + fn + 1e-8)
        no_collision_precision = tn / (tn + fn + 1e-8)
        no_collision_recall = tn / (tn + fp + 1e-8)
        
        # Calculate final macro metrics
        final_precision = (collision_precision + no_collision_precision) / 2
        final_recall = (collision_recall + no_collision_recall) / 2
        final_f1 = 2 * (final_precision * final_recall) / (final_precision + final_recall + 1e-8)
        
        print(f"\nOptimal threshold: {optimal_threshold:.4f}")
        print(f"Accuracy with optimal threshold: {accuracy:.4f}")
        print(f"Best macro F1 score: {final_f1:.4f}")
        print(f"Collision metrics - Precision: {collision_precision:.4f}, Recall: {collision_recall:.4f}")
        print(f"No Collision metrics - Precision: {no_collision_precision:.4f}, Recall: {no_collision_recall:.4f}")
        
        # Calculate APS score
        aps_score = sklearn.metrics.average_precision_score(all_labels, all_predictions)
        
        # Calculate collision F1
        collision_f1 = 2 * (collision_precision * collision_recall) / (collision_precision + collision_recall + 1e-8)
        
        # Return metrics using optimal threshold
        class_metrics = {
            'collision': {
                'precision': collision_precision, 
                'recall': collision_recall,
                'average_precision_score': aps_score,
                'f1': final_f1
            },
            'no_collision': {
                'precision': no_collision_precision, 
                'recall': no_collision_recall
            },
            'accuracy': accuracy
        }
        
        return avg_loss, np.array(all_predictions), accuracy, final_precision, final_recall, final_f1, aps_score, optimal_threshold, class_metrics

    def _save_checkpoint(self, epoch: int, metric_str: str):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }
        
        # Only add scheduler state if it exists
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        torch.save(checkpoint, f'checkpoints_newmodel/del025weight_epoch_{epoch}_{metric_str}.pt')

    def _print_metrics(self, epoch: int, train_loss: float, train_metrics: tuple, 
                      val_loss: float, val_metrics: tuple):
        train_acc, train_precision, train_recall, train_f1 = train_metrics
        val_acc, val_precision, val_recall, val_f1, val_aps, class_metrics = val_metrics
        
        print("\nEpoch Summary:")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Train Accuracy: {train_acc*100:.2f}%")
        print(f"Train Precision: {train_precision*100:.2f}%")
        print(f"Train Recall: {train_recall*100:.2f}%")
        print(f"Train F1: {train_f1*100:.2f}%")
        print(f"Val Loss: {val_loss:.4f}")
        print(f"Val Accuracy: {val_acc*100:.2f}%")
        print(f"Val Precision: {val_precision*100:.2f}%")
        print(f"Val Recall: {val_recall*100:.2f}%")
        print(f"Val F1: {val_f1*100:.2f}%")
        print(f"Val APS Score: {val_aps:.4f}")
        print("\nClass-specific metrics:")
        print(f"Collision - Precision: {class_metrics['collision']['precision']*100:.2f}%, "
              f"Recall: {class_metrics['collision']['recall']*100:.2f}%")
        print(f"No Collision - Precision: {class_metrics['no_collision']['precision']*100:.2f}%, "
              f"Recall: {class_metrics['no_collision']['recall']*100:.2f}%")
        print("-" * 50)


# def train_epoch(model, dataloader, criterion, optimizer, device):
#     model.train()
#     total_loss = 0
    
#     for batch_idx, (videos, labels) in enumerate(dataloader):
#         videos = videos.to(device)
#         labels = labels.to(device)
        
#         optimizer.zero_grad()
#         outputs = model(videos)
#         loss = criterion(outputs, labels)
        
#         loss.backward()
#         optimizer.step()
        
#         total_loss += loss.item()
        
#     return total_loss / len(dataloader)

# def validate(model, dataloader, criterion, device):
#     model.eval()
#     total_loss = 0
#     predictions = []
#     targets = []
    
#     with torch.no_grad():
#         for videos, labels in dataloader:
#             videos = videos.to(device)
#             labels = labels.to(device)
            
#             # Ensure labels have shape [batch_size, 1]
#             if len(labels.shape) == 1:
#                 labels = labels.view(-1, 1)
            
#             outputs = model(videos)
#             loss = criterion(outputs, labels)
            
#             total_loss += loss.item()
#             predictions.extend(outputs.cpu().numpy().flatten())  # Flatten predictions
#             targets.extend(labels.cpu().numpy().flatten())      # Flatten targets
    
#     return total_loss / len(dataloader), predictions, targets 