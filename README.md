# 変換方法 例

```
cd mmsegmentation
python pidnet_mmseg_inference.py \
    --config ./configs/pidnet/pidnet-l_2xb6-120k_1024x1024-cityscapes.py \
    --checkpoint /root/weights/pidnet/pytorch/cityscapes/pidnet-l_2xb6-120k_1024x1024-cityscapes_20230303_114514-0783ca6b.pth \
    --image-path /root/model_converter/samples/frankfurt_000000_003025_leftImg8bit.png \
    --output-path /root/experiments/exp_20251017/pidnet_mmseg_l.png \
    --export-torchscript /root/experiments/exp_20251017/pidnet_mmseg_l.pt
```
