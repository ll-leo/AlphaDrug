import torch
import time
import os
import shutil
import numpy as np
import random as rd
import argparse
from loguru import logger
from rdkit import Chem

from model.Lmser_Transformerr import MFT as DrugTransformer
# from model.Transformer import MFT as DrugTransformer
# from model.Transformer_Encoder import MFT as DrugTransformer

from utils.docking import CaculateAffinity, ProteinParser
from utils.log import timeLable, readSettings, VisualizeMCTS, saveMCTSRes, VisualizeInterMCTS
from beamsearch import sample
import selfies as sf


QE = 9
QMIN = QE
QMAX = QE
groundIndex = 0 # MCTS Node唯一计数
infoma = {}

class Node:

    def __init__(self, parentNode=None, childNodes=[], path=[],p=1.0, selMaxLen=999):
        global groundIndex
        self.index = groundIndex
        groundIndex += 1

        self.parentNode = parentNode
        self.childNodes = childNodes
        self.wins = 0
        self.visits = 0
        self.path = path  #MCTS 路径
        self.p = p
        self.selMaxLen = selMaxLen

    def SelectNode(self):
        nodeStatus = self.checkExpand()
        if nodeStatus == 4:
            puct = []
            for childNode in self.childNodes:
                puct.append(childNode.CaculatePUCT())
            
            m = np.max(puct)
            indices = np.nonzero(puct == m)[0]
            ind=rd.choice(indices)
            return self.childNodes[ind], self.childNodes[ind].checkExpand()
        
        return self, nodeStatus

    def AddNode(self, content, p):
        n = Node(self, [], self.path + [content], p=p, selMaxLen=self.selMaxLen)
        self.childNodes.append(n)
        return n
    
    def UpdateNode(self, wins):
        self.visits += 1
        self.wins += wins
        
    def CaculatePUCT(self):
        if not self.parentNode:
            return 0.0 # 画图用的
        # c = 1.5
        c = 1.5 if self.visits > 10 else 2.5 # 访问次数少时倾向探索
        if QMAX == QMIN:
            wins = 0
        else:
            if self.visits:
                wins = (self.wins/self.visits - QMIN) / (QMAX - QMIN)
            else: 
                wins = (QE - QMIN) / (QMAX - QMIN)
        
        return wins + c*self.p*np.sqrt(self.parentNode.visits)/(1+self.visits)
        # return wins/self.visits+50*self.p*np.sqrt(self.parentNode.visits)/(1+self.visits)
    
    def checkExpand(self):
        """
            node status: 1 terminal; 2 too long; 3 legal leaf node; 4 legal noleaf node
        """

        if self.path[-1] == '$':
            return 1
        elif not (len(self.path) < self.selMaxLen):
            return 2
        elif len(self.childNodes) == 0:
            return 3
        return 4
        
def JudgePath(path, selMaxLen):
    return (path[-1] != '$') and (len(path) < selMaxLen)

def Select(rootNode):
    while True:
        rootNode, nodeStatus = rootNode.SelectNode()
        if nodeStatus != 4:
            return rootNode, nodeStatus
  
# def Expand(rootNode, atomList, plist):
#     if JudgePath(rootNode.path, rootNode.selMaxLen):
#         for i, atom in enumerate(atomList):
#             rootNode.AddNode(atom, plist[i])
        
def Expand(rootNode, atomList, plist):
    if JudgePath(rootNode.path, rootNode.selMaxLen):
        k = 5  # 仅扩展概率最高的前 k 个分子
        top_indices = np.argsort(plist)[-k:]
        for i in top_indices:
            rootNode.AddNode(atomList[i], plist[i])


def Update(node, wins):
    while node:
        node.UpdateNode(wins)
        node = node.parentNode

def updateMinMax(node):
    # muzero method
    global QMIN
    global QMAX
    if node.visits:
        QMAX = max(QMAX, node.wins/node.visits)
        QMIN = min(QMIN, node.wins/node.visits)
        for child in node.childNodes:
            updateMinMax(child)

def rollout(node, model):
    path = node.path[:]
    selMaxLen = node.selMaxLen
    
    allScore = []
    allValidselfies = []
    allSelfies = []
    while JudgePath(path, selMaxLen):
        # 快速走子
        atomListExpanded, pListExpanded = sample(model, path, vocabulary, proVoc, selMaxLen, proMaxLen, device, 30, protein_seq)
        if not pListExpanded:
            # 处理空的 pListExpanded，例如跳过当前迭代或终止循环
            break
        
        m = np.max(pListExpanded)
        indices = np.nonzero(pListExpanded == m)[0]
        ind=rd.choice(indices)
        path.append(atomListExpanded[ind])
    # if path[-1] == '$':
    #     # selfiesK = ''.join([f'[{atom}]' for atom in path[1:-1]])
    #     selfiesK = ''.join(path[1:-1])
    #     allSelfies.append(selfiesK)
    #     try:
    #         selfiesK = sf.decoder(selfiesK)
    #         mols = Chem.MolFromSmiles(selfiesK)
    #     except:
    #         pass
    #     if mols and len(selfiesK) < selMaxLen:
    #         global infoma
    #         if selfiesK in infoma:
    #             affinity = infoma[selfiesK]
    #         else:
    #             affinity = CaculateAffinity(selfiesK, file_protein=pro_file[args.k], file_lig_ref=ligand_file[args.k], out_path=resFolderPath)
    #             infoma[selfiesK] = affinity
            
    #         if affinity == 500:
    #             Update(node, QMIN)
    #         else:
    #             logger.success(selfiesK + '       ' + str(-affinity))
    #             Update(node, -affinity)
    #             allScore.append(-affinity)
    #             allValidselfies.append(selfiesK)
    #     else:
    #         # logger.error(f"invalid: {''.join([f'[{atom}]' for atom in path])}")
    #         logger.error(f"invalid: {''.join(path)}")
    #         Update(node, QMIN)
    # else:
    #     # logger.warning(f"Abnormal ending: {''.join([f'[{atom}]' for atom in path])}")
    #     logger.warning(f"Abnormal ending: {''.join(path)}")
    #     Update(node, QMIN)
    
    if path[0] == '&':
        path = path[1:]
    else:
        path = path

    if path[-1] == '$':
        selfiesK = ''.join([f'[{atom}]' for atom in path[0:-1]])
        # selfiesK = ''.join(path[0:-1])
    else:
        selfiesK = ''.join([f'[{atom}]' for atom in path[0:]])
        # selfiesK = ''.join(path[0:])
    allSelfies.append(selfiesK)
    try:
        # 尝试将 SELFIES 解码为 SMILES
        decoded_smiles = sf.decoder(selfiesK)
        logger.info(f"Decoded SELFIES: {decoded_smiles}")
        selfiesK = decoded_smiles  # 将解码后的 SELFIES 替换为 SMILES
    except sf.DecoderError:
        logger.info(f"Assuming input is SMILES: {selfiesK}")

    # 验证 SMILES 的有效性
    mols = Chem.MolFromSmiles(selfiesK)
    if mols and len(selfiesK) < selMaxLen:
        global infoma
        # 检查是否已计算亲和力
        if selfiesK in infoma:
            affinity = infoma[selfiesK]
        else:
            # 计算亲和力
            affinity = CaculateAffinity(
                selfiesK, 
                file_protein=pro_file[args.k], 
                file_lig_ref=ligand_file[args.k], 
                out_path=resFolderPath
            )
            infoma[selfiesK] = affinity

        # 处理特殊情况的亲和力
        if affinity == 500:
            Update(node, QMIN)
        else:
            logger.success(f"Valid molecule: {selfiesK} with affinity: {-affinity}")
            Update(node, -affinity)
            allScore.append(-affinity)
            allValidselfies.append(selfiesK)
    else:
        # 无效的分子，记录日志并更新节点
        # logger.error(f"Invalid molecule: {''.join(path)}")
        logger.error(f"Invalid molecule: {''.join([f'[{atom}]' for atom in path[0:-1]])}")
        Update(node, QMIN)

    return allScore, allValidselfies, allSelfies
    
def MCTS(rootNode):
    allScore = []
    allValidSelfies = []
    allSelfies = []
    currSimulationTimes = 0
    
    while currSimulationTimes < simulation_times:
        
        global QMIN
        global QMAX
        QMIN = QE
        QMAX = QE
        updateMinMax(rootNode)
        currSimulationTimes += 1
        
        #MCTS SELECT
        node, _ = Select(rootNode)
        # VisualizeInterMCTS(rootNode, modelName, './', times, QMAX, QMIN, QE)

        #rollout
        score, validSelfies, aSelfies = rollout(node, model)
        allScore.extend(score)
        allValidSelfies.extend(validSelfies)
        allSelfies.extend(aSelfies)

        #MCTS EXPAND 
        atomList, logpListExpanded = sample(model, node.path, vocabulary, proVoc, selMaxLen, proMaxLen, device, 30, protein_seq)
        # pListExpanded = [np.exp(p) for p in logpListExpanded]

        # 加入温度缩放
        temperature = 0.8
        pListExpanded = [np.exp(p / temperature) for p in logpListExpanded]
        
        Expand(node, atomList, pListExpanded)

        
    if args.max:
        indices = np.argmax([n.visits for n in rootNode.childNodes])
    else:
        allvisit = np.sum([n.visits for n in rootNode.childNodes]) * 1.0
        prList = np.random.multinomial(1, [(n.visits)/allvisit for n in rootNode.childNodes], 1)
        indices = list(set(np.argmax(prList, axis=1)))[0]
        logger.info([(n.visits)/allvisit for n in rootNode.childNodes])

    return rootNode.childNodes[indices], allScore, allValidSelfies, allSelfies

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-k', type=int, default=0, help='protein index')
    parser.add_argument('--device', type=str, default='0')
    parser.add_argument('-st', type=int, default=10, help='simulation times')
    parser.add_argument('--source', type=str, default='new')
    parser.add_argument('-p', type=str, default='test3', help='pretrained model')

    parser.add_argument('--max', action="store_true", help='max mode')

    args = parser.parse_args()

    if args.source == 'new':
        test_pdblist = sorted(os.listdir('./data/test_pdbs/'))
        pro_file = ['./data/test_pdbs/%s/%s_protein.pdb'%(pdb,pdb) for pdb in test_pdblist]
        ligand_file = ['./data/test_pdbs/%s/%s_ligand.sdf'%(pdb,pdb) for pdb in test_pdblist]
        protein_seq = ProteinParser(test_pdblist[args.k])
    
    
    else:
        raise NotImplementedError('Unknown source: %s' % args.source)


    simulation_times = args.st
    experimentId = os.path.join('experiment', args.p)
    ST = time.time()

    modelName = '30.pt'
    hpc_device = "mps" if torch.backends.mps.is_available() else "cpu"
    mode = "max" if args.max else "freq"
    resFolder = '%s_%s_mcts_%s_%s_%s_%s_%s'%(hpc_device,mode,simulation_times, timeLable(), modelName, args.k, test_pdblist[args.k])

    resFolderPath = os.path.join(experimentId, resFolder)
    
    if not os.path.isdir(resFolderPath):
        os.mkdir(resFolderPath)
    logger.add(os.path.join(experimentId, resFolder, "{time}.log"))
    
    shutil.copyfile('./mcts.py',os.path.join(experimentId, resFolder) + '/mcts.py')
    
    
    if len(protein_seq) > 999:
        logger.info('skipping %s'%test_pdblist[args.k])
    else:
        
        s = readSettings(experimentId)
        vocabulary = s.selVoc
        proVoc = s.proVoc
        selMaxLen = int(s.selMaxLen)
        proMaxLen = int(s.proMaxLen)
        
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        device_ids = [0]  # 在 Mac 上多卡配置无效，保留单个设备

        model = DrugTransformer(**s)
        # model = torch.nn.DataParallel(model, device_ids=device_ids) # 指定要用到的设备
        model = model.to(device) # 模型加载到设备0
        model.load_state_dict(torch.load(experimentId +'/model/'+ modelName, map_location=device))
        model.to(device)
        model.eval()
        
        node = Node(path=['&'], selMaxLen=selMaxLen)
        
        times = 0
        allScores = []
        allValidSelfies = []
        allSelfies = []

        while(JudgePath(node.path, selMaxLen)):
            
            times += 1
            node, scores, validSelfies, selfies = MCTS(node)
            
            allScores.append(scores)
            allValidSelfies.append(validSelfies)
            allSelfies.append(selfies)

            VisualizeMCTS(node.parentNode, modelName, resFolderPath, times)

        alphaSel = ''
        affinity = 500
        if node.path[0] == '&':
            node.path = node.path[1:]
        else:
            node.path = node.path
        if node.path[-1] == '$':
            # alphaSel = ''.join(node.path[1:-1])  # 确保拼接到最后一个字符之前
            alphaSel = ''.join([f'[{atom}]' for atom in node.path[1:-1]])
        else:
            # alphaSel = ''.join(node.path[1:])    # 如果未以 `$` 结尾，直接拼接
            alphaSel = ''.join([f'[{atom}]' for atom in node.path[1:]])
        
        # alphaSel = '&' + alphaSel + '$'
        # alphaSel = sf.decoder(alphaSel) # selfies to smiles
        try:
            smi = sf.decoder(alphaSel)  # 尝试将 SELFIES 转换为 SMILES
            logger.info(f"Decoded SELFIES to SMILES: {smi}")
        except sf.DecoderError:
            # 如果不是 SELFIES，直接假定是 SMILES
            smi = alphaSel
        
        # 检查 SMILES 是否有效
        if Chem.MolFromSmiles(smi):
            logger.success(f"Valid SMILES: {smi}")
            
            # 计算亲和力（affinity）
            if smi in infoma:
                affinity = infoma[smi]
            else:
                affinity = CaculateAffinity(smi, file_protein=pro_file[args.k], file_lig_ref=ligand_file[args.k], out_path=resFolderPath)
            
            logger.success(f"Affinity: {-affinity}")
        else:
            logger.error(f"Invalid molecule: {alphaSel} (after decoding: {smi})")


        saveMCTSRes(resFolderPath, {
                'score': allScores,
                'allValidSelfies': allValidSelfies,
                'allSelfies': allSelfies,
                'finalSelfies': alphaSel,
                'finalScore': -affinity
            })

    ET = time.time()
    logger.info('time {}'.format((ET-ST)//60))


