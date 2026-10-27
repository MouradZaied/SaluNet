import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import copy
import math
import os
import sys
import time
import wandb

# Historical SaluNet-C-50/ImageNet-1K recipe.
use_mixup = True
use_cutmix = True
FP16 = True
CLIP_GRAD = 0.5
EMA_DECAY_BASE = 0.9999

BATCH_SIZE = 180  
ACCUMULATION_STEPS = 5  
NUM_WORKERS = 12

BASE_LR = 0.4
REFERENCE_BS = 1024
WEIGHT_DECAY = 2e-05
NORM_WEIGHT_DECAY = 0.0
MOMENTUM = 0.9

NUM_EPOCHS = 150
WARMUP_EPOCHS = 10

LABEL_SMOOTHING = 0.1
# The historical run used the same Beta distribution for MixUp and CutMix.
# Keeping one explicit value avoids suggesting that CutMix used alpha=1.0.
MIXING_ALPHA = 0.2

TRAIN_CROP_SIZE = 224
VAL_RESIZE_SIZE = 256
RA_REPS = 4
RANDOM_ERASE_PROB = 0.5

world_size = int(os.environ.get('WORLD_SIZE', 1))
EFFECTIVE_BS = BATCH_SIZE * world_size * ACCUMULATION_STEPS
LR = BASE_LR * (EFFECTIVE_BS / REFERENCE_BS)  

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


# CUDA settings used for the reported single-A40 run.
os.environ["CUDA_LAUNCH_BLOCKING"] = "0"
os.environ["TORCH_CUDNN_V8_API_ENABLED"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
torch.use_deterministic_algorithms(False)

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.enabled = True

try:
    torch.set_float32_matmul_precision('high')
except Exception:
    pass

print(f"PyTorch: {torch.__version__}")
print(f"    CUDA: {torch.version.cuda}")
print(f"    Device: {torch.cuda.get_device_name(0)}")
print(f"    GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
print(f"    TF32 enabled: {torch.backends.cuda.matmul.allow_tf32}")
print(f"    cuDNN v8: {os.environ.get('TORCH_CUDNN_V8_API_ENABLED', 'no')}")
print(f"    cuBLAS workspace: {os.environ.get('CUBLAS_WORKSPACE_CONFIG', 'undefined')}")
wandb.init(project="salunet-c50-imagenet1k")

class SALU(nn.Module):
    """Saturating activation with positive log-parameterized coefficients."""

    def __init__(self, a_init=1.0, b_init=0.0001, name=""):
        super().__init__()
        self.loga = nn.Parameter(torch.tensor(math.log(a_init)))
        self.logb = nn.Parameter(torch.tensor(math.log(b_init)))
        self.name = name

    def forward(self, x):
        x = x.float()
        a = torch.exp(self.loga)
        b = torch.exp(self.logb)
        denom = torch.sqrt(1.0 + a * b * x.square())
        return (a * x) / denom
    
    def get_params(self):
        """Return effective coefficients for experiment logging."""
        a = torch.exp(self.loga).item()
        b = torch.exp(self.logb).item()
        return {
            f"salu/{self.name}/a": a,
            f"salu/{self.name}/b": b
        }

class ClampedSWALU(nn.Module):
    """Historical SWALU variant with an output clamp to [-10, 10]."""

    def __init__(self, a=1.0, b=1.0, clamp_value=10.0, name=""):
        super().__init__()
        self.salu = SALU(a_init=a, b_init=b, name=name)
        self.clamp_value = clamp_value
        self.name = name

    def forward(self, x):
        out = 0.5 * x * (1 + self.salu(x))
        return torch.clamp(out, min=-self.clamp_value, max=self.clamp_value)

    def get_params(self):
        return self.salu.get_params()

# ================= MODEL =================
    
def build_model_with_controlled_downsample(bn_downsample_config, default_a=1.0, default_b=0.1):
    """Build the exact historical ResNet-50 with SALU and clamped SWALU."""
    model = models.resnet50(weights=None)
    salu_count = 0
    bn_kept_count = 0
    galu_count = 0
    bn_current_index = 0
    
    def process_batchnorm(module, name, child):
        nonlocal salu_count, bn_kept_count, bn_current_index
        bn_current_index += 1
        
        if bn_downsample_config and bn_current_index in bn_downsample_config:
            config = bn_downsample_config[bn_current_index]
            
            if config.get('replace', False):
                a_init = config.get('a', default_a)
                b_init = config.get('b', default_b)
                salu_count += 1
                setattr(module, name, SALU(
                    a_init=a_init, 
                    b_init=b_init, 
                    name=f"salu_downsample_{bn_current_index}"
                ))
                print(f"  BatchNorm #{bn_current_index} -> SALU (a={a_init}, b={b_init})")
            else:
                bn_kept_count += 1
                print(f"  BatchNorm #{bn_current_index} kept unchanged")
        else:
            salu_count += 1
            setattr(module, name, SALU(
                a_init=default_a, 
                b_init=default_b, 
                name=f"salu_bn_{bn_current_index}"
            ))
    
    def replace_batchnorm_by_index(module):
        for name, child in module.named_children():
            if isinstance(child, nn.BatchNorm2d):
                process_batchnorm(module, name, child)
            else:
                replace_batchnorm_by_index(child)
    
    replace_batchnorm_by_index(model)
    
    swalu_count = 0

    def replace_all_relu(module):
        nonlocal swalu_count
        for name, child in module.named_children():
            if isinstance(child, nn.ReLU):
                swalu_count += 1
                setattr(module, name, ClampedSWALU(name=f"swalu_{swalu_count}"))
            else:
                replace_all_relu(child)
    
    replace_all_relu(model)
    
    if (salu_count, bn_kept_count, swalu_count) != (53, 0, 17):
        raise RuntimeError(
            "Unexpected architecture: "
            f"{salu_count} SALU, {bn_kept_count} BatchNorm, "
            f"{swalu_count} clamped SWALU; expected 53, 0, 17."
        )

    print("\nArchitecture summary:")
    print(f"  - {salu_count} BatchNorm2d modules replaced by SALU")
    print(f"  - {swalu_count} ReLU modules replaced by clamped SWALU")
    
    return model

# ================= ACTIVATION PARAMETER COLLECTION =================
def collect_activation_params(model):
    """Collect effective SALU/SWALU parameters without debug overhead."""
    params = {}
    for module in model.modules():
        if hasattr(module, 'get_params'):
            try:
                params.update(module.get_params())
            except Exception:
                pass
    return params

# ================= PARAMETER GROUPS =================
def get_parameter_groups(model, base_lr, base_wd, norm_wd=0.0):
    normal_params = []
    norm_params = []
    special_params = []
   
    for name, param in model.named_parameters():
        if 'salu' in name.lower() or 'loga' in name or 'logb' in name:
            special_params.append(param)
        elif 'bn' in name or 'norm' in name or 'running_mean' in name or 'running_var' in name:
            norm_params.append(param)
        else:
            normal_params.append(param)
   
    param_groups = [
        {'params': normal_params, 'lr': base_lr, 'weight_decay': base_wd},
        {'params': norm_params, 'lr': base_lr, 'weight_decay': norm_wd},
        {'params': special_params, 'lr': base_lr * 2, 'weight_decay': 0.0}
    ]
   
    print("\nParameter groups:")
    print(f"  - Regular weights: {len(normal_params)} tensors (LR={base_lr:.6f}, WD={base_wd})")
    print(f"  - Norm layers: {len(norm_params)} tensors (LR={base_lr:.6f}, WD={norm_wd})")
    print(f"  - SALU/clamped SWALU: {len(special_params)} tensors (LR={base_lr*2:.6f}, WD=0)")
   
    return param_groups

# ================= EMA =================
class ModelEMA:
    def __init__(self, model, decay=0.9999):
        self.ema_model = copy.deepcopy(model)
        self.ema_model.eval()
        self.decay = decay
        self.num_updates = 0

    @torch.no_grad()
    def update(self, model):
        self.num_updates += 1
        decay = min(self.decay, (1 + self.num_updates) / (10 + self.num_updates))
        for ema_v, model_v in zip(self.ema_model.state_dict().values(), model.state_dict().values()):
            ema_v.copy_(ema_v * decay + model_v * (1. - decay))

# ================= REPEATED AUGMENTATION SAMPLER =================
class RepeatedAugmentationSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, data_source, num_repeats=1, shuffle=True):
        self.data_source = data_source
        self.num_repeats = num_repeats
        self.shuffle = shuffle
       
    def __iter__(self):
        n = len(self.data_source)
        indices = torch.arange(n).repeat(self.num_repeats)
        if self.shuffle:
            indices = indices[torch.randperm(len(indices))]
        return iter(indices.tolist())
   
    def __len__(self):
        return len(self.data_source) * self.num_repeats

# ================= DATA AUGMENTATION =================
from torchvision.transforms import TrivialAugmentWide

# Standard ImageNet statistics for ResNet.
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]

normalize = transforms.Normalize(mean=NORM_MEAN, std=NORM_STD)

# ImageNet transforms used by the historical run.
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(TRAIN_CROP_SIZE),  # 176
    transforms.RandomHorizontalFlip(),
    TrivialAugmentWide(num_magnitude_bins=31),      
    transforms.ToTensor(),
    normalize,
    transforms.RandomErasing(p=RANDOM_ERASE_PROB)   
])

# Validation transform.
val_transform = transforms.Compose([
    transforms.Resize(VAL_RESIZE_SIZE),  # 232
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize,
])

# Dataset
train_dataset = ImageFolder('/root/imagenet/train', transform=train_transform)
val_dataset = ImageFolder('/root/imagenet/val', transform=val_transform)

# Repeated augmentation sampler. RA_REPS=4 means four samples per image.
train_sampler = RepeatedAugmentationSampler(train_dataset, num_repeats=RA_REPS, shuffle=True)

# ============================================================
# Data-loader prefetching and deterministic worker initialization.
# ============================================================
def worker_init_fn(worker_id):
    # Seed each data-loader worker.
    seed = torch.initial_seed() % 2**32
    import numpy as np
    np.random.seed(seed + worker_id)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    sampler=train_sampler,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=4,
    worker_init_fn=worker_init_fn
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=4
)

# ================= CRITICAL SALU INITIALIZATION =================
# These five index-specific values are part of the reported recipe. Removing
# them caused the network to remain near loss 6.9 and prevented learning.
CRITICAL_SALU_INITIALIZATION = {
    1: {'replace': True, 'a': 5.5, 'b': 1e-5},
    5: {'replace': True, 'a': 2.0, 'b': 0.1},
    15: {'replace': True, 'a': 1.5, 'b': 0.01},
    28: {'replace': True, 'a': 0.8, 'b': 0.001},
    47: {'replace': True, 'a': 1.0, 'b': 0.001},
}

model = build_model_with_controlled_downsample(
    bn_downsample_config=CRITICAL_SALU_INITIALIZATION,
    default_a=1.0,
    default_b=0.1
).to(DEVICE)

model = model.to(memory_format=torch.channels_last)

#print(model)

param_groups = get_parameter_groups(model, LR, WEIGHT_DECAY, NORM_WEIGHT_DECAY)

# SGD optimizer.
optimizer = optim.SGD(param_groups, lr=LR, momentum=MOMENTUM)

#scaler = torch.amp.GradScaler('cuda')

"""Setup gradient scaler for FP16 training"""
try:
    # PyTorch >= 2.0
    scaler = torch.amp.GradScaler('cuda')
except AttributeError:
    # PyTorch < 2.0
    scaler = torch.cuda.amp.GradScaler()


ema = ModelEMA(model, decay=EMA_DECAY_BASE) if True else None  # use_ema = True
criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)

# Compile mode used by the historical run.
if hasattr(torch, 'compile'):
    model = torch.compile(model, mode="max-autotune-no-cudagraphs")

# ================= LR SCHEDULER COSINE ANNEALING =================
def adjust_learning_rate(epoch, step, steps_per_epoch):
    current_step = epoch * steps_per_epoch + step
    total_steps = NUM_EPOCHS * steps_per_epoch
    warmup_steps = WARMUP_EPOCHS * steps_per_epoch

    if current_step < warmup_steps:
        lr_scale = current_step / warmup_steps
    else:
        progress = (current_step - warmup_steps) / (total_steps - warmup_steps)
        lr_scale = 0.5 * (1 + math.cos(math.pi * progress))

    for i, param_group in enumerate(optimizer.param_groups):
        if i == 2:  # SALU and clamped SWALU parameters.
            param_group['lr'] = LR * 2 * lr_scale
        else:
            param_group['lr'] = LR * lr_scale
   
    return LR * lr_scale

# ================= MIXUP / CUTMIX =================
def apply_mix(images, targets):
    lam = torch.distributions.Beta(MIXING_ALPHA, MIXING_ALPHA).sample().item()
    index = torch.randperm(images.size(0)).to(DEVICE)

    # Preserve the historical random-number order: lambda and the permutation
    # are sampled before choosing between CutMix and MixUp.
    if use_cutmix and torch.rand(1).item() > 0.5:
        W, H = images.size(2), images.size(3)
        cut_rat = math.sqrt(1. - lam)
        cut_w, cut_h = int(W * cut_rat), int(H * cut_rat)
        cx, cy = torch.randint(W, (1,)).item(), torch.randint(H, (1,)).item()
        x1, y1 = max(cx - cut_w // 2, 0), max(cy - cut_h // 2, 0)
        x2, y2 = min(cx + cut_w // 2, W), min(cy + cut_h // 2, H)
        images[:, :, x1:x2, y1:y2] = images[index, :, x1:x2, y1:y2]
        lam = 1 - ((x2 - x1) * (y2 - y1) / (W * H))
    else:
        images = lam * images + (1 - lam) * images[index]

    return images, targets, targets[index], lam

# ================= TRAIN =================
def train_one_epoch(epoch):
    model.train()
    start_time = time.time()
    running_loss = 0.0
    correct = 0
    total = 0
    optimizer.zero_grad()

    for batch_idx, (images, targets) in enumerate(train_loader):   
        lr_base = adjust_learning_rate(epoch, batch_idx, len(train_loader))
        
        images = images.to(DEVICE, non_blocking=True).contiguous().to(memory_format=torch.channels_last)
        targets = targets.to(DEVICE, non_blocking=True)

        if use_mixup or use_cutmix:
            images, ta, tb, lam = apply_mix(images, targets)

        with torch.amp.autocast('cuda', enabled=FP16):
            outputs = model(images)
            if use_mixup or use_cutmix:
                loss = lam * criterion(outputs, ta) + (1 - lam) * criterion(outputs, tb)
            else:
                loss = criterion(outputs, targets)
            loss = loss / ACCUMULATION_STEPS

        scaler.scale(loss).backward()

        if (batch_idx + 1) % ACCUMULATION_STEPS == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_GRAD)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            if ema:
                ema.update(model)

        running_loss += loss.item() * ACCUMULATION_STEPS * images.size(0)
        # Historical diagnostic only: with MixUp/CutMix this compares predictions
        # against the original hard target and is not a mixed-label accuracy.
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        elapsed = time.time() - start_time
        est_total = elapsed / (batch_idx + 1) * len(train_loader)
       
        sys.stdout.write(
            f"\rEpoch [{epoch+1}/{NUM_EPOCHS}] "
            f"LR: {lr_base:.6f} | "
            f"Loss: {running_loss/total:.4f} | "
            f"Acc: {100.*correct/total:.2f}% | "
            f"ETA: {(est_total-elapsed)/60:.1f}m"
        )
        sys.stdout.flush()

    print()
    return running_loss/total, 100.*correct/total, lr_base

# ================= EVAL =================
def evaluate(model, loader):
    model.eval()
    correct1 = correct5 = total = 0
    total_loss = 0.0

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(DEVICE, non_blocking=True).contiguous().to(memory_format=torch.channels_last)
            targets = targets.to(DEVICE, non_blocking=True)
            outputs = model(images)
            loss = criterion(outputs, targets)
            total_loss += loss.item() * images.size(0)
            _, pred1 = outputs.topk(1, 1, True, True)
            _, pred5 = outputs.topk(5, 1, True, True)
            total += targets.size(0)
            correct1 += pred1.eq(targets.view(-1, 1)).sum().item()
            correct5 += pred5.eq(targets.view(-1, 1).expand_as(pred5)).sum().item()

    return total_loss/total, 100.*correct1/total, 100.*correct5/total

# ================= CHECKPOINTING =================
best_checkpoint_path = None

def save_checkpoint(epoch, model_to_save, optimizer, scheduler, scaler, best_top1, is_best=False):
    """Save the best complete EMA checkpoint."""
    if not is_best:
        return
    
    os.makedirs("checkpoints", exist_ok=True)
    
    global best_checkpoint_path
    if best_checkpoint_path is not None and os.path.exists(best_checkpoint_path):
        os.remove(best_checkpoint_path)
        print(f"  Removed previous best checkpoint: {best_checkpoint_path}")
    
    filename = os.path.join("checkpoints", f"best_checkpoint_epoch{epoch}_top{best_top1:.2f}.pth")
    best_checkpoint_path = filename
    
    checkpoint = {
        'epoch': epoch,
        'best_top1': best_top1,
        'model_state_dict': model_to_save.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
    }
    torch.save(checkpoint, filename)
    print(f"  Saved complete checkpoint: {filename} (Top-1: {best_top1:.2f}%)")

# ================= TRAINING LOOP =================
print(f"Starting training on {DEVICE} with {world_size} GPUs")
print(f"Batch size per GPU: {BATCH_SIZE}, Effective batch size: {EFFECTIVE_BS}")
print(f"Learning rate: {LR:.6f} (SGD + momentum={MOMENTUM})")
print(f"Epochs: {NUM_EPOCHS}, Warmup: {WARMUP_EPOCHS}")
print(f"Weight decay: normal={WEIGHT_DECAY}, norm={NORM_WEIGHT_DECAY}")
print("Special rules: SALU/clamped SWALU -> LR x2, WD=0")
print(f"Repeated Augmentation: {RA_REPS} repeat(s)")
print(f"Random Erasing: p={RANDOM_ERASE_PROB}")
print(f"Train crop: {TRAIN_CROP_SIZE}, Val resize: {VAL_RESIZE_SIZE}")
print(f"CutMix: {'enabled' if use_cutmix else 'disabled'}")
print("Automatic augmentation: TrivialAugmentWide")
print(f"MixUp/CutMix shared alpha: {MIXING_ALPHA}")
print("-" * 80)

best_top1_ema = 0.0

for epoch in range(NUM_EPOCHS):
    train_loss, train_acc, current_lr = train_one_epoch(epoch)
    val_loss, top1, top5 = evaluate(model, val_loader)
   
    if ema:
        val_loss_ema, top1_ema, top5_ema = evaluate(ema.ema_model, val_loader)
    else:
        top1_ema = top1
   
    # Collect activation parameters only once, at the end of training.
    if epoch == NUM_EPOCHS - 1:
        activation_params = collect_activation_params(model)
        torch.save(activation_params, 'final_salu_swalu_params.pt')
        print("\nSaved final SALU/clamped-SWALU parameters to "
              "'final_salu_swalu_params.pt'")
    else:
        activation_params = {}
   
    log_dict = {
        "train/loss": train_loss,
        "train/acc": train_acc,
        "train/lr_base": current_lr,
        "val/loss": val_loss,
        "val/top1": top1,
        "val/top5": top5,
        "epoch": epoch + 1,
    }
   
    if ema:
        log_dict.update({
            "val_ema/top1": top1_ema,
            "val_ema/top5": top5_ema,
        })
   
    # Limit the final activation-parameter sample to avoid W&B overhead.
    if activation_params:
        sample_params = {k: v for i, (k, v) in enumerate(activation_params.items()) if i < 10}
        log_dict.update(sample_params)
    
    wandb.log(log_dict, step=epoch)
   
    print(f"\n{'='*60}")
    print(f"Epoch {epoch+1}/{NUM_EPOCHS} - Summary")
    print(f"{'='*60}")
    print(f"TRAIN | LR: {current_lr:.6f} | Loss: {train_loss:.4f} | Top1: {train_acc:.2f}%")
    print(f"VAL   | Loss: {val_loss:.4f} | Top1: {top1:.2f}% | Top5: {top5:.2f}%")
    if ema:
        print(f"VAL_EMA | Top1: {top1_ema:.2f}% | Top5: {top5_ema:.2f}%")
   
    # Save the best EMA model.
    if ema and top1_ema > best_top1_ema:
        best_top1_ema = top1_ema
        save_checkpoint(
            epoch=epoch+1, 
            model_to_save=ema.ema_model,
            optimizer=optimizer, 
            scheduler=None,
            scaler=scaler, 
            best_top1=best_top1_ema, 
            is_best=True
        )
        print("  New best EMA score")
   
    # Display a small final activation-parameter sample.
    if activation_params and epoch == NUM_EPOCHS - 1:
        sample_keys = list(activation_params.keys())[:4]
        print(f"\nSample activation parameters (final):")
        for key in sample_keys:
            print(f"  {key}: {activation_params[key]:.4f}")
        if len(activation_params) > 4:
            print(f"  ... and {len(activation_params)-4} more")

print(f"\n{'='*60}")
print("Training completed!")
print(f"Best EMA Top-1: {best_top1_ema:.2f}%")
print(f"Best checkpoint saved: {best_checkpoint_path}")
print("Final SALU/clamped-SWALU parameters saved to "
      "'final_salu_swalu_params.pt'")
print(f"{'='*60}")

wandb.finish()
