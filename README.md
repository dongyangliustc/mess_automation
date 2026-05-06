# MESS自动化工具包 - 从量子化学计算生成MESS输入文件

## 🚀 快速开始

### 1. 安装依赖
```bash
# 进入项目目录
cd mess_automation

# 安装Python依赖
pip install -r requirements.txt
```

**最小依赖要求**：
- Python 3.8+
- PyYAML>=6.0
- Jinja2>=3.0.0
- numpy>=1.21.0 (可选，用于高级处理)

### 2. 准备高斯输出文件
确保您有以下类型的高斯输出文件：
- **反应物 (Reactant)**: 优化的几何结构 + 频率计算
- **过渡态 (Transition State)**: 优化的过渡态结构 + 频率计算 (包含虚频率)
- **产物 (Product)**: 优化的产物结构 + 频率计算

所有文件应为高斯16输出格式 (`.out` 或 `.log`)。

### 3. 创建配置文件
创建一个YAML配置文件，例如 `my_reaction.yaml`:

```yaml
# 输入/输出设置
input:
  files:
    - "path/to/reactant.out"
    - "path/to/ts.out"
    - "path/to/product.out"

output:
  mess_input: "my_reaction_mess.inp"
  log_file: "processing.log"

# 量子化学设置
quantum:
  frequency_scaling_factor: 0.971  # 频率标度因子
  geometry_units: "angstrom"      # 几何单位：angstrom 或 bohr
  frequency_units: "1/cm"         # 频率单位：1/cm 或 Hz
  energy_units: "kcal/mol"        # 能量单位：kcal/mol, kJ/mol, hartree

# MESS全局设置
mess_global:
  temperature_list: [300, 400, 500, 600, 700, 800, 900, 1000]  # 温度列表(K)
  pressure_list: [1.0]                                         # 压力列表(atm)
  calculation_method: "direct"                                 # 计算方法：direct 或 microcanonical

# 反应网络定义
reaction_network:
  species:
    - name: "Reactant"
      type: "well"              # 类型：well, bimolecular, barrier
      gaussian_file: "path/to/reactant.out"
      symmetry_factor: 0.5      # 对称因子
      ground_energy: 0.0        # 相对能量 (kcal/mol)
    
    - name: "TS"
      type: "barrier"
      gaussian_file: "path/to/ts.out"
      symmetry_factor: 0.5
      zero_energy: 10.0         # 势垒高度 (kcal/mol)
      connects: ["Reactant", "Product"]  # 连接的物种
    
    - name: "Product"
      type: "well"
      gaussian_file: "path/to/product.out"
      symmetry_factor: 0.5
      ground_energy: -5.0       # 相对产物能量 (kcal/mol)

# 处理选项
processing:
  skip_unconverged: true        # 跳过未收敛的计算
  validate_frequencies: true    # 验证频率数据
  create_backups: false         # 创建备份文件
  verbose: true                 # 详细输出
```

### 4. 运行工具
```bash
# 基本用法
python main.py -c my_reaction.yaml -o mess_output.inp

# 使用自定义频率标度因子 (覆盖配置文件)
python main.py -c my_reaction.yaml -o mess_output.inp --scaling 0.967

# 详细输出模式
python main.py -c my_reaction.yaml -o mess_output.inp -v

# 覆盖已存在的输出文件
python main.py -c my_reaction.yaml -o mess_output.inp --overwrite
```

### 5. 检查输出
工具将生成：
- `mess_output.inp`: 完整的MESS输入文件
- `processing.log`: 处理日志 (如配置)
- 控制台输出显示处理的物种和统计数据

## 📋 详细使用步骤

### 第一步：环境准备
1. **确保Python版本**：检查Python版本 >= 3.8
   ```bash
   python --version
   ```

2. **安装依赖**：
   ```bash
   pip install PyYAML Jinja2 numpy scipy
   ```
   或使用提供的requirements.txt
   ```bash
   pip install -r requirements.txt
   ```

### 第二步：组织文件结构
建议的文件结构：
```
my_project/
├── gaussian_outputs/
│   ├── reactant.out
│   ├── ts.out
│   └── product.out
├── config/
│   └── reaction_config.yaml
├── scripts/
│   └── run_mess_automation.py
└── outputs/
    ├── mess_input.inp
    └── processing.log
```

### 第三步：创建配置文件
使用提供的模板或复制 `examples/test_config.yaml` 并修改：
```bash
# 复制示例配置文件
cp examples/test_config.yaml my_reaction.yaml
# 编辑文件路径和参数
```

### 第四步：运行验证测试
```bash
# 使用内置示例测试工具是否正常工作
cd mess_automation
python main.py -c examples/test_config.yaml -o test_output.inp -v
```

预期输出：
- ✅ 成功解析3个高斯输出文件
- ✅ 提取几何结构 (16原子)
- ✅ 提取频率数据 (42频率)
- ✅ 检测过渡态虚频率并处理
- ✅ 生成MESS输入文件 (约200行)

### 第五步：生成MESS输入文件
```bash
# 为您的反应生成MESS输入
python main.py -c my_reaction.yaml -o my_mess_input.inp -v
```

### 第六步：验证生成的MESS文件
检查生成的MESS文件：
1. **确保所有物种正确包含**
2. **检查频率数量正确** (3N-6 或 3N-5)
3. **确认势垒连接正确**
4. **验证能量单位一致**

## 🔧 高级用法

### 多步反应网络
对于复杂的反应网络，可以在YAML配置中定义多个物种和势垒：

```yaml
reaction_network:
  species:
    - name: "A"
      type: "well"
      gaussian_file: "A.out"
      symmetry_factor: 1.0
      ground_energy: 0.0
    
    - name: "B"
      type: "well"
      gaussian_file: "B.out"
      symmetry_factor: 1.0
      ground_energy: -2.5
    
    - name: "C"
      type: "well"
      gaussian_file: "C.out"
      symmetry_factor: 2.0
      ground_energy: -5.0
    
    - name: "TS1"
      type: "barrier"
      gaussian_file: "TS1.out"
      symmetry_factor: 1.0
      zero_energy: 8.0
      connects: ["A", "B"]
    
    - name: "TS2"
      type: "barrier"
      gaussian_file: "TS2.out"
      symmetry_factor: 1.0
      zero_energy: 6.0
      connects: ["B", "C"]
```

### 程序化使用
您也可以在Python脚本中直接使用模块：

```python
import sys
import os
sys.path.append("/path/to/mess_automation")

from parser import GaussianParser
from processor import FrequencyCorrector
from assembler import MESSAssembler
import yaml

# 1. 解析高斯输出
parser = GaussianParser(skip_unconverged=True)
quantum_data = parser.parse_file("molecule.out")

# 2. 应用频率校正
corrector = FrequencyCorrector(scaling_factor=0.971)
correction = corrector.correct_frequencies(quantum_data)

# 3. 创建MESS汇编器
assembler = MESSAssembler(template_dir="templates")

# 4. 配置全局设置
assembler.global_settings = {
    "temperature_list": [300, 400, 500, 600],
    "pressure_list": [1.0],
    "calculation_method": "direct"
}

# 5. 添加物种
assembler.add_species(
    name="molecule",
    quantum_data=quantum_data,
    correction=correction,
    species_type="well",
    symmetry_factor=1.0
)

# 6. 生成MESS输入
assembler.write_to_file("output.inp")
```

### 命令行选项详解
```bash
python main.py --help
```
输出：
```
usage: main.py [-h] -c CONFIG -o OUTPUT [--scaling SCALING] [--overwrite] [-v]

MESS Automation Tool - Generate MESS input files from quantum chemistry data

options:
  -h, --help           show this help message and exit
  -c, --config CONFIG  Path to configuration YAML file
  -o, --output OUTPUT  Path to output MESS input file
  --scaling SCALING    Frequency scaling factor (overrides config)
  --overwrite          Overwrite output file if it exists
  -v, --verbose        Verbose output

Examples:
  # Generate MESS input with default configuration
  python main.py -c config.yaml -o mess_input.inp
  
  # Use custom scaling factor
  python main.py -c config.yaml -o mess_input.inp --scaling 0.971
  
  # Verbose output
  python main.py -c config.yaml -o mess_input.inp -v
  
  # Overwrite existing output file
  python main.py -c config.yaml -o mess_input.inp --overwrite
```

## 📁 项目结构

```
mess_automation/
├── parser.py              # 高斯输出解析器
├── processor.py           # 频率校正和单位转换
├── assembler.py           # MESS输入文件组装器 (使用Jinja2模板)
├── main.py               # 命令行主程序
├── config/
│   ├── template_config.yaml  # 配置文件模板
│   └── advanced_config.yaml  # 高级配置示例
├── templates/            # Jinja2模板目录
│   ├── global_section.jinja2    # MESS全局设置模板
│   ├── model_section.jinja2     # 模型部分模板
│   ├── species.jinja2           # 物种模板
│   └── barrier.jinja2           # 势垒模板
├── examples/             # 示例文件
│   ├── test_config.yaml         # 测试配置文件
│   └── example_data/
│       ├── reactant.out         # 示例反应物
│       ├── ts.out              # 示例过渡态
│       └── product.out         # 示例产物
├── tests/                # 单元测试
├── requirements.txt      # Python依赖
└── README.md            # 本文档
```

## 🔍 故障排除

### 常见问题

#### 1. YAML配置文件错误
**症状**: `Error parsing YAML configuration: while scanning a simple key`
**解决方案**: 确保YAML语法正确，缩进一致，冒号后要有空格
```yaml
# 正确
zero_energy: 10.0

# 错误
ZeroEnergy[kcal/mol]    10.0  # 无效的YAML语法
```

#### 2. 高斯文件解析失败
**症状**: `Failed to parse Gaussian output: Could not find Standard orientation`
**解决方案**: 
- 确保高斯计算已完成并正常终止
- 检查输出文件中是否有"Normal termination"消息
- 对于优化计算，确认有"Optimization completed"

#### 3. 频率提取错误
**症状**: `Incorrect number of frequencies extracted`
**解决方案**:
- 确保是频率计算作业 (`freq` 关键词)
- 检查是否包含虚频率 (过渡态应有1个虚频率)
- 确认分子不是线性 (线性分子频率数应为3N-5)

#### 4. 原子符号映射错误
**症状**: 原子符号显示不正确 (如C显示为H)
**解决方案**: 此bug已在最新版本修复，确保使用更新后的parser.py

### 调试模式
使用详细输出模式查看详细处理过程：
```bash
python main.py -c config.yaml -o output.inp -v
```

这将显示：
- 每个文件的解析状态
- 提取的原子和频率数量
- 应用的单位转换
- 生成的MESS文件统计

## 📊 示例：完整工作流程

### 示例1：简单反应 A → B
1. **准备高斯文件**: `A.out` (反应物), `TS.out` (过渡态), `B.out` (产物)
2. **创建配置文件** `simple_reaction.yaml`:
   ```yaml
   input:
     files: ["A.out", "TS.out", "B.out"]
   output:
     mess_input: "simple_mess.inp"
   quantum:
     frequency_scaling_factor: 0.971
   mess_global:
     temperature_list: [298, 400, 500]
     pressure_list: [1.0]
   reaction_network:
     species:
       - name: "A"
         type: "well"
         gaussian_file: "A.out"
         symmetry_factor: 1.0
         ground_energy: 0.0
       - name: "TS"
         type: "barrier"
         gaussian_file: "TS.out"
         symmetry_factor: 1.0
         zero_energy: 15.0
         connects: ["A", "B"]
       - name: "B"
         type: "well"
         gaussian_file: "B.out"
         symmetry_factor: 1.0
         ground_energy: -10.0
   ```
3. **运行工具**:
   ```bash
   python main.py -c simple_reaction.yaml -o simple_mess.inp -v
   ```

### 示例2：测试包内置示例
```bash
cd mess_automation
python main.py -c examples/test_config.yaml -o test_output.inp --overwrite -v
```

## 🔬 技术细节

### 支持的量子化学程序
- **主要支持**: Gaussian 16 (输出文件格式)
- **理论支持**: 其他量子化学程序 (需自定义解析器)

### 提取的数据
1. **几何结构**: 笛卡尔坐标 (Å)
2. **振动频率**: 谐波频率 (cm⁻¹)，自动处理虚频率
3. **电子能量**: SCF能量 (Hartree)
4. **零点能**: ZPE (从频率计算)
5. **热化学数据**: (如配置)

### 频率处理
- 应用标度因子 (默认: 0.971)
- 虚频率转为正值用于MESS
- 单位转换 (cm⁻¹ ↔ Hz)

### MESS文件格式
生成的MESS文件包含：
1. **全局部分**: 温度、压力、计算方法
2. **模型部分**: 能量转移模型参数
3. **物种部分**: 几何、频率、对称因子
4. **势垒部分**: 连接信息、势垒高度

## 🤝 贡献

欢迎贡献！请遵循以下步骤：

1. Fork项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 打开Pull Request

### 开发设置
```bash
# 克隆仓库
git clone https://github.com/yourusername/mess-automation.git
cd mess-automation

# 安装开发依赖
pip install -r requirements.txt
pip install pytest pytest-cov black flake8 mypy

# 运行测试
pytest tests/

# 检查代码风格
black --check .
flake8 .
mypy .
```

## 📄 许可证

MIT License - 详见LICENSE文件

## 📞 支持

如遇问题：
1. 检查故障排除章节
2. 运行测试用例确认工具工作正常
3. 提交Issue并提供：
   - 配置文件内容
   - 错误消息
   - 高斯文件片段 (前50行)
   - Python版本和环境信息

---

**版本**: 1.0.0  
**最后更新**: 2026年4月29日  
**作者**: MESS自动化开发团队