# 练老师工具集 - 图像LLM反推功能

这个ComfyUI自定义节点工具集提供了使用GLM免费大语言模型对图像进行反推以及视频提示词优化的功能。

## 功能介绍

- **ImageToLLMReverseNode**: 核心节点，读取图像并使用GLM模型对图像进行反推
- **VideoPromptOptimizerNode**: 视频提示词优化节点，专门用于优化和扩写视频相关提示词
- **LLMAPIKeyManagerNode**: API密钥管理节点，用于保存和清除GLM API密钥配置
- 可自定义反推要求、模型参数等

## 安装方法

1. 将此工具集文件夹放入ComfyUI的`custom_nodes`目录下
2. 安装必要的依赖：
   ```bash
   pip install -r requirements.txt
   ```
3. 重启ComfyUI

## 使用方法

### 图像反推节点
1. 在ComfyUI中找到`lian GLM图像反推节点`
2. 连接图像输入
3. 输入API密钥（如已保存，可留空使用已保存的密钥）
4. 输入反推要求（提示词）
5. 调整其他可选参数（如模型名称、温度等）
6. 运行工作流

### 视频提示词优化节点
1. 在ComfyUI中找到`lian 视频提示词优化节点`
2. 输入需要优化的视频提示词（单一输入）
3. 输入API密钥（如已保存，可留空使用已保存的密钥）
4. 调整优化强度和其他可选参数
5. 运行工作流获取优化后的通义万相格式提示词
6. 可以将优化后的提示词连接到其他节点使用

### API密钥管理节点
1. 在ComfyUI中找到`lian LLM API密钥管理`节点
2. 输入GLM API密钥
3. 选择操作类型：
   - **save**：保存API密钥到配置文件
   - **clear**：清除已保存的API密钥
4. 运行工作流，查看操作状态

## 节点参数说明

### ImageToLLMReverseNode

- **images**: 输入图像
- **reverse_prompt**: 反推要求，描述你希望模型如何分析图像
- **api_key**: GLM模型API密钥（留空则使用已保存的密钥）
- **model_name**: 可选，指定具体GLM模型名称
- **temperature**: 温度参数，控制输出的随机性
- **max_tokens**: 最大生成 tokens 数

### VideoPromptOptimizerNode

- **input_prompt**: 输入提示词，需要优化的视频相关描述
- **api_key**: GLM模型API密钥（留空则使用已保存的密钥）
- **model_name**: 可选，指定具体GLM模型名称
- **temperature**: 温度参数，控制输出的随机性
- **max_tokens**: 最大生成 tokens 数
- **optimization_strength**: 优化强度（0.5-2.0），控制扩写程度

该节点会按照专业的视频提示词六要素结构自动优化和扩写输入提示词，输出纯提示词内容，不包含任何额外解释。优化结构包括：主体描述（详细刻画主要对象或人物的外观、特征、状态）、场景描述（细致描绘主体所处环境，包括时间、地点、背景元素、光线、天气等）、运动描述（明确主体的动作细节）、镜头语言（指定景别、视角、镜头类型、运镜方式）、氛围词（定义画面的情感与气氛）、风格化（设定画面的艺术风格）。

### LLMAPIKeyManagerNode

- **api_key**: GLM API密钥
- **action**: 操作类型，支持save（保存）和clear（清除）

## 配置文件说明

API密钥会保存在以下位置：
- 配置目录：`ComfyUI/custom_nodes/lianlaoshi_tools/config/`
- 密钥文件：`config/api_keys.json`
- 环境变量文件：`config/.env`

## 注意事项

1. 使用前需要获取GLM模型的API密钥
2. GLM模型需要从智谱AI获取API密钥
3. 确保网络连接正常
4. 大型图像可能会增加处理时间和API调用成本
5. 配置文件包含敏感信息，请妥善保管
6. 如使用环境变量，可以在系统环境变量中设置，优先级高于配置文件

## 更新日志

- 优化节点: VideoPromptOptimizerNode升级为通义万相专用格式，只接收单一提示词输入
- 格式规范: 按照通义万相视频提示词格式（主体+场景+运动+美学控制+风格化）进行优化
- 纯提示词输出: 优化结果只包含提示词内容，不包含任何额外解释
- 分离功能: 将图像反推和视频提示词优化功能分离，各自独立节点
- 新增节点: 添加VideoPromptOptimizerNode专门用于视频提示词优化
- 优化版本: 仅使用GLM模型，移除千问模型支持，简化代码结构