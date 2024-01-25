基于华为MindSpore AI框架开发小型GPT模型，实现金庸风格小说续写。

由于MindSpore大部分逻辑和Pytorch相近，因此将其视为加深对GPT理解的小型项目
+ 在原本代码的基础上尝试优化修改，提升了推理速度。
+ 修改数据预处理和训练逻辑，允许加载预训练参数、分阶段训练。

+ 环境：
  + 训练/调试环境：启智AI平台。
    + 代码托管：https://openi.pcl.ac.cn/drf_dolphin/gpt
    + 镜像：MindSpore2.0.0a0_cann_6.0.rc1.alpha005
    + 资源规格：NPU: 1*Ascend 910, CPU: 24, 显存: 32GB, 内存: 256GB
  + 推理环境：华为云。
    + modelarts的mindspore2.2+cann7.0.1
+ 数据集：金庸小说，以 “鸳鸯刀.txt”为训练样本，vocab_size=1928。
+ 实现逻辑参考：https://github.com/karpathy/nanoGPT

这个GPT的问题：
+ 输入的prompt并不支持训练集以外的任何token，否则会报错。
+ GPT本身的输出具有随机性，不对随机性进行限制导致整体输出结果语义语法不通顺。
  + 一开始版本只能输出一些符号。
  + 初次成功训练的版本GPT_1-112_10.ckpt可以输出正常的文字，但是几乎没有正常的一句话。
    + 限制每次取的最高概率字符数，TOPK=5。
    + 优化随机选取策略使其具有更好的稳定性。
    + 成功使其输出正常的一句话，但是每句话之间关联还是比较少。依然存在不少的无意义语句。
+ 在MindSpore2.0.0下推理速度非常慢，但是在MindSpore2.2下推理比较快。
  + 是MindSpore版本问题，代码中可优化空间小。
+ 输出达到限制之后就停止，无论一句话是否结束。

发现：
+ 在src/gpt.py的CausalSelfAttention类中，调用MindSpore官方API`att = F.masked_fill(att, mask, -1e9)`效率并不如直接进行mask计算来的快。