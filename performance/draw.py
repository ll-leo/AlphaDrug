from rdkit import Chem
from rdkit.Chem import Draw
import random
import json
from PIL import Image

# 加载 JSON 数据
file_path = "experiment/test3/mps_freq_mcts_10_2024_12_11_15_26_20_30.pt_0_1a9u/res.json"
with open(file_path, 'r') as f:
    data = json.load(f)

# 展平嵌套数据
nested_valid_smiles = data['allValidSelfies']
flat_valid_smiles = [smiles for sublist in nested_valid_smiles for smiles in sublist]

# 随机选择一些分子进行可视化
num_molecules_to_visualize = 6  # 要展示的分子数
random_smiles = random.sample(flat_valid_smiles, min(num_molecules_to_visualize, len(flat_valid_smiles)))

# 转换为 RDKit 分子对象
molecules = [Chem.MolFromSmiles(smiles) for smiles in random_smiles if Chem.MolFromSmiles(smiles) is not None]

# 添加分子标签（显示 SMILES）
legends = [f"SMILES: {smiles}" for smiles in random_smiles if Chem.MolFromSmiles(smiles) is not None]

# 可视化分子（放大子图，显示标签）
img = Draw.MolsToGridImage(
    molecules,
    molsPerRow=3,  # 每行显示 3 个分子
    subImgSize=(300, 300),  # 每个分子子图大小
    legends=legends,  # 显示分子标签
)

# 保存为文件
output_path = "molecule_visualization_labeled.png"
img.save(output_path)

# 使用 PIL 查看保存的图像
img_pil = Image.open(output_path)
img_pil.show()