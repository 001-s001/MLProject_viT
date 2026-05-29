# ViT PPT 大纲

## 1. ViT 核心思想

- 把图像切成 patch，把 patch 当成 token。
- 使用标准 Transformer encoder 建模图像全局关系。
- 与 CNN 不同，ViT 的局部性不是手工写入结构，而是更多依赖数据学习。

## 2. ViT 结构图与公式

- Patch embedding: `x_p -> z_0`
- Add class token and position embedding.
- Transformer block: `MSA + MLP`
- Classification head uses final class token.

## 3. 复现实验设置

- Dataset: CIFAR-10, CIFAR-100.
- Model: SimpleViT, `embed_dim=192`, `depth=6`, `num_heads=3`.
- Optimizer: AdamW.
- Main configs: patch size 4 and patch size 8.

## 4. 结果与消融

- 表格：CIFAR-10 patch 4 vs patch 8。
- 表格：CIFAR-10 vs CIFAR-100。
- 可选：from scratch vs ImageNet pretrained.
- 解释：patch 越大 token 越少，计算更低，但细节损失更明显。

## 5. Failure Analysis

- 混淆矩阵展示最容易混淆的类别。
- 错误样本网格展示高置信度错误预测。
- 解释：小目标、背景干扰、局部纹理不足时 ViT 可能失败。

## 6. 与 ResNet / ConvNeXt V2 比较

- ResNet：强局部归纳偏置，适合小数据稳定训练。
- ViT：全局建模强，依赖大规模预训练。
- ConvNeXt V2：现代 CNN，试图结合 CNN 结构优势和现代训练策略。

