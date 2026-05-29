# ViT 论文笔记

论文：An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale

## 核心问题

ViT 想回答的问题是：如果不使用卷积网络的强局部归纳偏置，能否直接把 Transformer 用在图像分类上？论文把图像切成固定大小 patch，把每个 patch 当成类似 NLP token 的输入，再用标准 Transformer encoder 建模全局关系。

## 方法结构

- Image patching：把输入图像切成 `P x P` 的 patch。
- Patch embedding：每个 patch 展平成向量后线性投影到 `D` 维 token；在代码中可以用 `Conv2d(kernel_size=P, stride=P)` 等价实现。
- Class token：在 patch token 前拼接一个可学习 `[CLS]` token，用于最终分类。
- Position embedding：加入可学习位置编码，让模型知道 patch 的空间位置。
- Transformer encoder：由 multi-head self-attention 和 MLP block 堆叠而成。
- MLP head：取最终 `[CLS]` token 的输出做分类。

## 关键结论

ViT 在大规模预训练数据上表现很强，但在小数据集上从零训练时不一定优于 CNN。主要原因是 CNN 自带局部性和平移等变等归纳偏置，而 ViT 需要更多数据学习这些结构。

## 本项目复现边界

本项目不复现 JFT-300M 大规模预训练，而是在 CIFAR-10/100 上复现 ViT 的核心结构和训练行为。重点关注：

- ViT 小模型能否跑通并达到合理准确率。
- patch size 对性能和计算的影响。
- ViT 在小数据集上的失败模式。
- 与 ResNet、ConvNeXt V2 的架构差异。

## 与 ResNet / ConvNeXt V2 的关系

ResNet 代表经典 CNN 路线，通过残差连接解决深层网络优化问题。ViT 代表把 Transformer 引入视觉任务的架构转折点。ConvNeXt V2 则可以理解为 CNN 在 Transformer 时代的现代化回应：保留卷积结构，同时吸收大模型训练、归一化、自监督等经验。

