# Model Files

The pretrained model files are too large for GitHub (>100MB limit). Please download them from the following sources:

## FiDIP Pretrained Models
- **HRNet FiDIP Model**: `hrnet_fidip.pth` (243.53 MB)
  - Download from: [FiDIP_models](https://drive.google.com/drive/folders/108P-1SnTqaj3xNtjYZ1o7T8z6UvUYuiC?usp=sharing)
  - Place in: `models/hrnet_fidip.pth`

- **MobileNet FiDIP Model**: `mobile_fidip.pth` (16 MB)
  - Download from: [FiDIP_models](https://drive.google.com/drive/folders/108P-1SnTqaj3xNtjYZ1o7T8z6UvUYuiC?usp=sharing)
  - Place in: `models/mobile_fidip.pth`

## Standard Pretrained Models
- **HRNet COCO Model**: `w32_384×288.pth` (109.39 MB)
  - Download from: [TGA_models](https://drive.google.com/drive/folders/14kAA1zXuKODYgrRiQmKnVcipbY7RedVV)
  - Place in: `models/coco/w32_384×288.pth`

## Setup Instructions
1. Create the models directory structure:
   ```bash
   mkdir -p models/coco
   ```

2. Download the model files to their respective locations as listed above.

3. Verify the models are in place:
   ```bash
   ls -la models/
   ls -la models/coco/
   ```

## Model Usage
Once downloaded, you can use these models with the test scripts:
```bash
# Test HRNet FiDIP model
python tools/test_adaptive_model.py \
    --cfg experiments/coco/hrnet/w48_384x288_adam_lr1e-3_infant.yaml \
    TEST.MODEL_FILE models/hrnet_fidip.pth TEST.USE_GT_BBOX True

# Test MobileNet FiDIP model  
python tools/test_adaptive_model.py \
    --cfg experiments/coco/mobilenet/mobile_224x224_adam_lr1e-3_infant.yaml \
    TEST.MODEL_FILE models/mobile_fidip.pth TEST.USE_GT_BBOX True
```
