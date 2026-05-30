import torch
import torch.nn as nn
from pathlib import Path
import sys

# Add YOLOv5 to path
yolov5_path = Path(__file__).parent.parent / "yolov5"
sys.path.insert(0, str(yolov5_path))

from models.common import DetectMultiBackend
from models.yolo import Model
from utils.general import check_img_size
from utils.torch_utils import select_device

class YOLOv5sDetector(nn.Module):
    """
    YOLOv5s wrapper for federated learning.
    """
    def __init__(self, num_classes=10, img_size=640):
        super().__init__()
        self.num_classes = num_classes
        self.img_size = img_size
        
        # Load YOLOv5s configuration
        import yaml
        with open(Path(yolov5_path) / "models" / "yolov5s.yaml", 'r') as f:
            self.cfg = yaml.safe_load(f)
        
        # Update number of classes
        self.cfg['nc'] = num_classes
        
        # Initialize model
        self.model = Model(self.cfg, ch=3, nc=num_classes)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights."""
        for m in self.model.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """Forward pass."""
        return self.model(x)
    
    def get_weights(self):
        """Extract model weights for FL transmission."""
        return {k: v.cpu() for k, v in self.model.state_dict().items()}
    
    def set_weights(self, weights):
        """Set model weights from FL aggregation."""
        self.model.load_state_dict(weights, strict=False)

def get_model(num_classes=10, img_size=640):
    """Factory function to create YOLOv5s model."""
    return YOLOv5sDetector(num_classes, img_size)