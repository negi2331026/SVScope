# visualization脚本
# 新加功能：生成共识序列、注释保守/差异、特殊区间
# 需要先将目录切换到 /NAS/wg_tkl/PanCancer_TKL/PanCancer_somaticTandemRepeat/TDScopeV28/src/ ，再激活tkl环境使用

'''
Clearly visualization of TDscope
1, Reads Extraction, sort reads into vline plot, 
2, Reads Feature selection;
3, Reads Clustering 
4, Somatic event selection
'''
import os,re 
import pandas as pd 
import numpy as np
import pysam
import networkx as nx
## EM Clustering 
from DataScanner import *
from ReadsCluster import *
from DecisionMaker import *
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import matplotlib.pyplot as plt 
import time 
from scipy.stats import chi2_contingency
from matplotlib.gridspec import GridSpec
import matplotlib.patches as patches
from spoa import poa         # spoa version 0.2.1 
import logging
import argparse
import functools
from multiprocessing import Pool
np.random.seed(2023)


#  ************Settings************ # 
# 5色组合
base_colors = np.array(['#00a97a','#82C5FB', '#FFB74D', '#F06292'])
baseName = np.array(['A','T','C','G'])
gap_color = '#BDBDBD'
category_colors = ['#1F77B4', '#FD884E', '#55c79a', '#D62728', '#d497dc', '#A3AEE5','#d4d3ff', "#D681A9",'#ffab98','#ff98c1', '#ffce74']
ClusterDict = {"ref":0, 'normal':1, 'tumor':2}
Sample_colors = ['#007197','#008d97','#67b29f','#9add9e','#d7f5ad','#ffcf91','#ea985d','#adcdf5','#80deff','#6ef9cf', '#FF7F0E', '#1F77B4']
Sample_dict = {"T1":0, "T2":1, "T3":2, "T4":3, "T5":4, "T6":5, "T7":6, "N1":7, "N3":8, "N5":9, "WBC":10, "ref":11}
annotation_colors = ['#7ec2ec', '#ed7891','#00b9a3','#8790cf','#41b289','#f9f871','#5093db','#ea817d','#d8f4ee','#b1f992','#e4a8ce']


#  ************Functions************ # 

# Data Scanner 
def SeqEncoder(seqinput):
    alphabet = {'A':0, 'T': 1, 'C':2, 'G':3, '-':4}
    encodeList = []
    for s in seqinput:
        encodeList.append(alphabet[s.upper()])
    return(np.array(encodeList))

def SeqDecoder(seqinput):
    alphabet = {0:'A', 1:'T', 2:'C', 3:'G', 4:'-'}
    decodeseq = ''
    for s in seqinput:
        if s != 4:
            decodeseq += alphabet[s]
    return(decodeseq)

def SeqAligner(seqList):
    # Input selected sequence list 
    # return MSA result for these sequences
    consensus, msa = poa(seqList, 1)
    seqdatamx = np.array(list(map(SeqEncoder, msa)))
    return(seqdatamx)

def CallMargin_f5(msa, flank_5):
    # select TD start and end region for further analysis 
    ## check the hg38 reference for msa columns setting 
    examplesequence = msa[0]
    IDXPool = []
    tmpflank = ''
    for I in range(len(examplesequence)):
        if examplesequence[I] !="-":
            tmpflank += examplesequence[I]
            IDXPool.append(I)
        if tmpflank == flank_5:
            break
    return(IDXPool)

def CallMargin_f3(msa, flank_3):
    # select TD start and end region for further analysis 
    ## check the hg38 reference for msa columns setting 
    examplesequence = msa[0]
    IDXPool = []
    tmpflank = ''
    for I in range(len(examplesequence)-1, 0, -1):
        if examplesequence[I] !="-":
            tmpflank = examplesequence[I] + tmpflank
            IDXPool.append(I)
        if tmpflank == flank_3:
            break
    return(IDXPool)

def CallMargin(msa,flank_5,flank_3):
    # select TD start and end region for further analysis 
    ## check the hg38 reference for msa columns setting 
    examplesequence = msa[0]
    IDXPool = []
    tmpflank = ''
    for I in range(len(examplesequence)):
        if examplesequence[I] !="-":
            tmpflank += examplesequence[I]
            IDXPool.append(I)
        if tmpflank == flank_5:
            break
    tmpflank = ''
    for I in range(len(examplesequence)-1, 0, -1):
        if examplesequence[I] !="-":
            tmpflank = examplesequence[I] + tmpflank
            IDXPool.append(I)
        if tmpflank == flank_3:
            break
    return(np.array(IDXPool))

def FindNonSameSite(seqencode_New_Sub, cutoff=3):
    featureExists = np.where(np.isnan(seqencode_New_Sub), 0, 1)
    TotalCount = featureExists.sum(axis=0)
    FeatureCount = []
    for Label in range(5):
        tmpFeatureCount = np.zeros((seqencode_New_Sub.shape[1],))
        ColIDX,Count = np.unique(np.where(seqencode_New_Sub==Label)[1], return_counts=True)
        for I in range(ColIDX.shape[0]):
            tmpFeatureCount[ColIDX[I]] = Count[I]
        FeatureCount.append(tmpFeatureCount)
    FeatureCountArr = np.array(FeatureCount)
    NonSameSiteIDX = np.where(np.sort(FeatureCountArr, axis=0)[-2] >= cutoff)[0]
    return(NonSameSiteIDX)

# Data Scanner 
# Df function Loading # 

def CountSomTD(Record,CA_type):
    # 卡方统计，判别高频TD是否来源于特定类型的肿瘤
    TypeName,TotalNum = np.unique(CA_type, return_counts=True)
    Type_TD = np.zeros((TypeName.shape[0],))
    TDType,TDCount = np.unique(CA_type[np.where(Record!=0)[0]], return_counts=True)
    for T in TDType:
        Type_TD[np.where(TypeName == T)[0]] += TDCount[np.where(TDType==T)[0]]
    Type_TD_Non = TotalNum - Type_TD
    return(np.vstack([Type_TD, Type_TD_Non]))


def CountPatient(WindowPatientCount, TotalPatientCount):
    # 计算各癌种的正负分布
    PosCount = np.zeros(TotalPatientCount.shape[1])
    for i in range(WindowPatientCount.shape[1]):
        CAtype, Count = WindowPatientCount[:,i]
        PosCount[np.where(TotalPatientCount[0,:]==CAtype)[0]] += Count
    NegCount = TotalPatientCount[1,:] - PosCount
    return(np.vstack([PosCount, NegCount]))


def R_OE(crossTab):
    # crossTab is a 2D numpy array
    # First row: number of positive cases for each cancer type
    # Second row: number of negative cases for each cancer type
    # Extract observed positive cases
    Observed = crossTab[0, :]
    # Total number of positive and negative cases across all cancer types
    total_positive = crossTab[0, :].sum()
    total_negative = crossTab[1, :].sum()
    total_cases = total_positive + total_negative
    # Calculate expected positive cases for each cancer type
    cancer_type_totals = crossTab.sum(axis=0)
    Expected = (cancer_type_totals * total_positive) / total_cases
    # Calculate R(O/E) for each cancer type
    ROE = Observed / Expected
    return ROE


def AnnoR_OE(Record, CA_type):
    TotalPatientCount = np.vstack(np.unique(CA_type,return_counts=True))
    PositivePatient = Record.loc[(Record==1)].index
    WindowPatientCount = np.vstack(np.unique(CA_type.loc[PositivePatient], return_counts=True))
    crossTab = CountPatient(WindowPatientCount, TotalPatientCount)
    ROE = R_OE(crossTab)
    chi2, pvalue, dof, ex = chi2_contingency(crossTab)
    return(np.array(list(ROE)+[pvalue]), crossTab)


def AnnoTDwindow(Record, CA_type):
    AnnoSTR = '-'
    crossTab = CountSomTD(Record,CA_type)
    ROE = R_OE(crossTab)
    TypeName,TotalNum = np.unique(CA_type, return_counts=True)
    chi2, pvalue, dof, ex = chi2_contingency(crossTab)
    enrichedType = np.array([])
    if pvalue <= 0.05:
        # enrichedType = TypeName[np.where((ROE>=1.5)&(crossTab[0,:]>=5))[0]]
        enrichedType = TypeName[np.where((ROE>=1.5)&(crossTab[0,:]>=4))[0]]
    if enrichedType.shape[0] > 0:
        AnnoSTR = ",".join(list(enrichedType))
    return(AnnoSTR)


def FetchTDsubSeq(TDRecord_to_find, savedir):
    npzFileList = [os.path.join(savedir, x) for x in os.listdir(savedir) if re.search('npz', x)]
    for npzFile in npzFileList:
        Dat = np.load(npzFile, allow_pickle=True)['DatSet']
        for I in range(Dat.shape[0]):
            sequenceList, ReadIDs, flank_5, flank_3, TDRecord = Dat[I]
            TDchrom, TDStart, TDEnd = TDRecord.strip().split("\t")[0],TDRecord.strip().split("\t")[1], TDRecord.strip().split("\t")[2]
            window = ','.join([TDchrom, TDStart, TDEnd])
            if window == TDRecord_to_find:
                found_record = (sequenceList, ReadIDs, flank_5, flank_3, TDRecord_to_find)
                readTDSeq = sequenceList
                readIDList = ReadIDs
                readIDList = readIDList.tolist()
                # print(len(readTDSeq), len(readIDList))
    return(readTDSeq, readIDList)


def SequencePlotRaw(ax, seqdatamx_Raw, readID, title='RawSeq',
                    base_colors=np.array(['#FFB74D', '#64B5F6', '#81C784', '#F06292']),
                    baseName = np.array(['A','T','C','G']), 
                    gap_color = '#999999', ClusterDict=ClusterDict, 
                    category_colors=category_colors):
    # Plot for raw extracted sequence 
    L = np.max([len(x) for x in seqdatamx_Raw])
    LabelX1,LabelX2, LabelYMIN, LabelYMAX, ClusterL, CAtypeL = [],[],[],[],[],[]
    PlotIdx = 0 
    print('SequencePlotRaw')
    for IDX in range(len(readID) ):
    # for IDX in [137]:
        Dat = SeqEncoder(np.array(list(seqdatamx_Raw[IDX])))
        if seqdatamx_Raw[IDX] == []:
            print(f"警告：seqdatamx_Raw[{IDX}] 处的编码数据为空，已跳过处理。")
            continue
        else:
            C = readID[IDX].split("|")[0].split('_')[-1]
            if np.where(Dat==4)[0].shape[0] > 0:
                ax.hlines(y=PlotIdx, xmin=0, xmax=Dat.shape[0], linewidth=0.5, linestyle='dashed', color=gap_color, zorder=-2)
            ax.vlines(x=np.where(Dat!=4)[0], ymin=PlotIdx-0.5, ymax=PlotIdx+0.5, color=np.array(base_colors)[Dat[np.where(Dat!=4)[0]]], zorder=-1)
            # ax.vlines(x=np.where(Dat!=4)[0], ymin=PlotIdx-0.5, ymax=PlotIdx+0.5, color=np.array(base_colors)[Dat[np.where(Dat!=4)[0]].astype(int)], zorder=-1)
            LabelX1.append(L+15)
            LabelX2.append(L+10)
            LabelYMIN.append(PlotIdx-0.5)
            LabelYMAX.append(PlotIdx+0.5)
            ClusterL.append(C)
            PlotIdx += -1
    LabelList = np.array([[i,color,label] for i, (color, label) in enumerate(zip(base_colors, baseName), start=1)], dtype=object)
    for i, (color, label) in enumerate(zip(base_colors, baseName), start=1):
        ax.vlines(x=i, ymin=PlotIdx-100.5, ymax=PlotIdx-99.5, color=color, label=label, zorder=3)
    for ClustI in np.unique(ClusterL):
        # print(ClustI)
        ax.vlines(x=ClusterDict[ClustI]+20, ymin=PlotIdx-200-0.5, ymax=PlotIdx-200+0.5, color=category_colors[ClusterDict[ClustI]], linewidth=3, zorder=3, label=ClustI)
    readLabelLen = 0.05 * L 
    for labelIDX in np.arange(len(LabelYMIN)):
        LabelL = LabelYMIN[labelIDX]
        ClustI = ClusterL[labelIDX]
        rect = patches.Rectangle((-readLabelLen, LabelL), readLabelLen-1, 1, linewidth=1, edgecolor=category_colors[ClusterDict[ClustI]], facecolor=category_colors[ClusterDict[ClustI]])
        ax.add_patch(rect)
    ax.set_xlim((-readLabelLen, L+1))
    ax.set_ylim((PlotIdx,1))
    ax.set_yticks([])
    ax.set_xticks([])
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.1), ncol=4)
    ax.set_title(title)
    return(0)


# 计算出现频率最大的数字，如果出现最多的是4（gap），那么就换成出现次数第二多的
def calculate_most_bp(cluster_seqs, idx):
    numbers = [seq[idx] for seq in cluster_seqs]
    counter = Counter(numbers)
    # 按出现次数从大到小排序
    sorted_items = sorted(counter.items(), key=lambda item: item[1], reverse=True)
    if sorted_items[0][0] == 4 and len(sorted_items) > 1:
        return sorted_items[1][0]
    return sorted_items[0][0]

    

# 针对每一列计算出非gap的出现频率最高的碱基
def GenerateConsensus(seqdatamx_Raw, readID, flank_5, flank_3, hcutoff = 3, scutoff=0.05):
    # Plot for filtered matrix 
    sequences = seqdatamx_Raw
    readLen = np.array([len(x) for x in sequences[1:]])
    DELIDX = np.where(readLen==0)[0]
    if DELIDX.shape[0] > 0:      # Fully DEL reads exist impute alignment matrix as fully gap 
        UnDELIDX = np.setdiff1d(np.arange(len(readID)), DELIDX)
        UnDELReads = [readID[I] for I in UnDELIDX]
        DELReads = [readID[I] for I in DELIDX]
        UnDELSeq = [sequences[I] for I in UnDELIDX]
        unconsensus, unmsa = poa(sequences, 1)
        unseqencode_New = list(map(SeqEncoder, unmsa))
        mxlen = len(unseqencode_New[-1])
        readID = np.array(UnDELReads + DELReads)
        seqencode_New = np.array(unseqencode_New + [[4] * mxlen] * len(DELReads))
        msa = unmsa + [["-"] * mxlen] * len(DELReads)
    else:
        consensus, msa = poa(sequences, 1)
        seqencode_New = np.array(list(map(SeqEncoder, msa)))
    # Remove the Non-associated flank sequence based on reference backbone 
    IDXPool = CallMargin(msa, flank_5, flank_3)
    seqdatamx_all = seqencode_New[1:]
    # readID = readID[1:]
    # 确定要保留的非侧翼序列索引
    non_flank_idx = np.setdiff1d(np.arange(seqdatamx_all.shape[1]), IDXPool)
    consencus = []
    for i in non_flank_idx:   
        most_bp = calculate_most_bp(seqdatamx_all, i)  
        consencus.append(most_bp)
    fasta = SeqDecoder(consencus)
    print(fasta)
    return (fasta)


def Main(sampleID, TDRecord, refFile,LabelList, saveDir, offset, mapQ, out_dir, consensus_output, fa_too_long):
    # Visualize Data 
    TDchrom, TDStart, TDEnd = TDRecord.strip().split(",")[0],int(TDRecord.strip().split(",")[1]), int(TDRecord.strip().split(",")[2])
    ref = pysam.FastaFile(refFile)
    refSeq = ref.fetch(TDchrom, TDStart-offset, TDEnd+offset).upper()
    flank_5, flank_3 = ref.fetch(TDchrom, TDStart-offset, TDStart).upper(), ref.fetch(TDchrom, TDEnd, TDEnd+offset).upper()
    readTDSeq, readIDList= FetchTDsubSeq(TDRecord, saveDir)
    readTDSeqNew = readTDSeq                                                # sub read selection uncertain flank aln reads would be removed
    ReadIDs = readIDList
    seqdatamx_Raw = [refSeq] + readTDSeqNew
    readID = ['ref|Seq'] + ReadIDs
    consensus, msa = poa(seqdatamx_Raw,1)
    fasta = GenerateConsensus(seqdatamx_Raw, readID, flank_5, flank_3, hcutoff = 3, scutoff=0.05)
    newline1 = f'>{sampleID}_{TDRecord}\n'
    newline2 = f'{fasta}\n'        
    if len(fasta) < 20000:
        with open(consensus_output, 'w') as output:
            output.write(newline1 + newline2)
    else:
        with open(fa_too_long, 'w') as fin:
            fin.write(newline1 + newline2)
        print('###################################################################################################################')
        print(fa_too_long)
        print(
            '###################################################################################################################')
        newline2 = fasta[0:10000] + '\n'
        with open(consensus_output, 'w') as output:
            output.write(newline1 + newline2)

###################### 生成consensus，跑循环
npz_dir = '/NAS/wg_tkl/PanCancer_TKL/PanCancer_somaticTandemRepeat/panCancer_Project_AllSV/HCC10'
out_dir = '/NAS/wg_lyc/project/Evolution/dataPrepare/insSearching/HCC10'
file = '/NAS/wg_lyc/project/Evolution/dataPrepare/insSearching/allSamples.HCC.multiple_somatic.Raw.bed'
with open(file, 'r') as input:
    for line in input:
        parts = line.strip().split('\t')
        sample = parts[0]
        if sample == 'HCC10':
            window = parts[1]
            TDRecord = ','.join(window.split('_'))
            consensus = out_dir + '/%s_srm2hg38_INS_sequence.fasta' % window
            fa_long = out_dir + '/%s_srm2hg38_INS_long_sequence.fasta' % window
            Main(patientID, TDRecord, Reference, LabelList, npz_dir,  offset, mapQ, out_dir, consensus,fa_long)  


# 计算每列中各碱基出现频数
def calculate_frequency(cluster_seqs, idx):
    num_seqs = len(cluster_seqs)
    # 初始化一个字典来存储每个索引位置上 0 - 4 出现的频数
    frequency_dict = {j: 0 for j in range(5)}
    # 统计当前索引位置上 0 - 4 出现的次数
    counts = {i: 0 for i in range(5)}
    for seq in cluster_seqs:
        num = seq[idx]
        counts[num] += 1
    # 计算频率
    for num in range(5):
        frequency_dict[num] = counts[num]
    return frequency_dict


def GetSpecialRegionIDX(input_file):
    df = pd.read_csv(input_file, sep='\s+', header=None, skiprows=2)
    df.columns = ['SW score', 'perc div.', 'perc del.', 'perc ins.', 'id', 'start', 'end', 
                  '(left)', '+','matching repeat', 'class/family', 'repeat start', 'repeat end', '(left).1', 'ID']
    print(df)
    filtered_df = df[~df['class/family'].isin(['Simple_repeat', 'Low_complexity', 'Satellite', 'Tandem_repeat'])]
    print(filtered_df)
    if not filtered_df.empty:
        start = filtered_df['start']
        end = filtered_df['end']
        category = filtered_df['class/family']
        if len(filtered_df) == 1:
            start = [start.item()]
            end = [end.item()]
            category = [category.item()]
        else:
            start = start.tolist()
            end = end.tolist()
            category = category.tolist()
    else:
        print("Special Region Not Found")
        start = None
        end = None
    return category, start, end


def SequencePlotCluster(ax, seqdatamx_Raw, readID, flank_5, flank_3, out,
                        title='RawSeq', hcutoff = 3, scutoff=0.05, 
                        base_colors=np.array(['#FFB74D', '#64B5F6', '#81C784', '#F06292']),
                        baseName = np.array(['A','T','C','G']), 
                        gap_color = '#C9C8C8', ClusterDict=ClusterDict, 
                        category_colors=category_colors, Sample_dict = Sample_dict, Sample_colors=Sample_colors,annotation_colors=annotation_colors):
    # Plot for filtered matrix 
    sequences = seqdatamx_Raw
    readLen = np.array([len(x) for x in sequences[1:]])
    DELIDX = np.where(readLen==0)[0]
    if DELIDX.shape[0] > 0:      # Fully DEL reads exist impute alignment matrix as fully gap 
        UnDELIDX = np.setdiff1d(np.arange(len(readID)), DELIDX)
        UnDELReads = [readID[I] for I in UnDELIDX]
        DELReads = [readID[I] for I in DELIDX]
        UnDELSeq = [sequences[I] for I in UnDELIDX]
        unconsensus, unmsa = poa(sequences, 1)
        unseqencode_New = list(map(SeqEncoder, unmsa))
        mxlen = len(unseqencode_New[-1])
        readID = np.array(UnDELReads + DELReads)
        seqencode_New = np.array(unseqencode_New + [[4] * mxlen] * len(DELReads))
        msa = unmsa + [["-"] * mxlen] * len(DELReads)
    else:
        consensus, msa = poa(sequences, 1)
        seqencode_New = np.array(list(map(SeqEncoder, msa)))
    # Remove the Non-associated flank sequence based on reference backbone
    seqdatamx_all = seqencode_New[1:]
    IDXPool = CallMargin(msa, flank_5, flank_3)
    NonSameIDX = np.setdiff1d(FindNonSameSite(seqencode_New, cutoff=max([hcutoff,seqdatamx_all.shape[0]*scutoff])), IDXPool)
    SamIDX = np.setdiff1d(np.setdiff1d(np.arange(seqdatamx_all.shape[1]), NonSameIDX), IDXPool)
    seqdatamx = seqencode_New[1:, NonSameIDX]
    # readID = readID[1:]
    K, seqdatamx, Rclust, thetap, gamma, pie, BICarray = EMCluster(seqdatamx)    
    non_flank_idx = np.setdiff1d(np.arange(seqdatamx_all.shape[1]), IDXPool)  # 确定要保留的非侧翼序列索引
    # print(non_flank_idx)
    raw_pvalues_from_tables = []
    # 先遍历每一列
    for i in non_flank_idx:   
        contingency_table = np.zeros((len(np.unique(Rclust)), 5))
        for X in np.unique(Rclust):
            seq_X = []
            for idx in np.where(Rclust==X)[0]:               
                seq = seqdatamx_all[idx]  
                seq_X.append(seq)
            result = calculate_frequency(seq_X, i)
            # print(result)
            for num in range(5):
                contingency_table[X, num] += result[num]
        # 计算每列的和
        column_sums = contingency_table.sum(axis=0)
        # 找到和不为 0 的列的索引
        non_zero_columns = np.where(column_sums != 0)[0]
        # 筛选出和不为 0 的列
        contingency_table = contingency_table[:, non_zero_columns]    
        # print(contingency_table)   
        chi2, pvalue, dof, ex = chi2_contingency(contingency_table)
        # print(chi2, pvalue, dof, ex)
        raw_pvalues_from_tables.append(pvalue)
    # 进行FDR校正
    reject_tables, fdr_pvals_tables, _, _ = multipletests(raw_pvalues_from_tables, alpha=0.05, method='fdr_bh')
    fdr_T_IDX = non_flank_idx[np.where(reject_tables)[0]].tolist()
    fdr_F_IDX = non_flank_idx[np.where(~reject_tables)[0]].tolist()
    # 获得特殊区间的IDX
    category, start, end = GetSpecialRegionIDX(out)
    # 将 category、start 和 end 对应起来
    category_dict = {}
    for cat, s, e in zip(category, start, end):
        if cat not in category_dict:
            category_dict[cat] = []
        category_dict[cat].append([s, e])
    # 对每个 category 的配对列表进行填充
    filled_category_dict = {}
    for cat, pairs in category_dict.items():
        filled_pairs = []
        for pair in pairs:
            if pair[1] - pair[0] > 1:
                # 若差值大于 1，进行填充
                fill = list(range(pair[0], pair[1] + 1))
                filled_pairs.extend(fill)
            else:
                # 若差值不大于 1，保持原样
                filled_pairs.extend(pair)
        filled_category_dict[cat] = filled_pairs
    # 将填充后的数值定位到 non_flank_idx 中
    Region_idx_dict = {}
    all_used_indices = []
    for cat, filled_list in filled_category_dict.items():
        Region_idx = [non_flank_idx[i - 1] for i in filled_list if i in non_flank_idx]
        Region_idx_dict[cat] = Region_idx
        all_used_indices.extend(Region_idx)
        print(cat,all_used_indices)
    # 找出 non_flank_idx 中没有定位到任何 category 的索引
    nonSpeRegionIDX = [idx for idx in non_flank_idx if idx not in all_used_indices]
    L = np.max([len(x) for x in seqdatamx_all])
    LabelX1,LabelX2, LabelYMIN, LabelYMAX, ClusterL, SampleL, CAtypeL = [],[],[],[],[],[],[]
    PlotIdx = 0 
    for X in np.unique(Rclust):
        category_tmp = category_colors[X]
        for IDX in np.where(Rclust==X)[0]:
            Dat = seqdatamx_all[IDX]
            # print(readID[IDX])
            C = readID[IDX].split("|")[0].split('_')[-1]
            if readID[IDX] == 'ref|Seq':
                S = 'ref'
            else:
                S = readID[IDX].split("|")[0].split('_')[1]
            if np.where(Dat==4)[0].shape[0] > 0:
                ax.hlines(y=PlotIdx, xmin=0, xmax=Dat.shape[0], linewidth=0.5, linestyle='dashed', color=gap_color, zorder=-2)
            ax.vlines(x=np.where(Dat!=4)[0], ymin=PlotIdx-0.5, ymax=PlotIdx+0.5, color=np.array(base_colors)[Dat[np.where(Dat!=4)[0]]], zorder=-1)
            LabelX1.append(L+15)
            LabelX2.append(L+10)
            LabelYMIN.append(PlotIdx-0.5)
            LabelYMAX.append(PlotIdx+0.5)
            ClusterL.append(C)
            SampleL.append(S)
            CAtypeL.append(X)
            PlotIdx += -1
    FeatureLen = seqdatamx.shape[0] * 0.1
    FeatureLabel = ['Common', 'Features', 'Flank']
    for i, IDXtmp in enumerate([SamIDX, NonSameIDX, IDXPool]):
        ax.vlines(x=IDXtmp, ymin=0, ymax=FeatureLen, color=category_colors[-1*i-1], label=FeatureLabel[i])
    DiffLabel = ['Conserve', 'Differential', 'Flank']
    for i, IDXtmp in enumerate([fdr_F_IDX, fdr_T_IDX, IDXPool]):
        ax.vlines(x=IDXtmp, ymin=FeatureLen, ymax=2*FeatureLen, color=annotation_colors[-1*i-1], label=DiffLabel[i])
    SpeRegionLabel = list(Region_idx_dict.keys()) + ['NonSpecial'] + ['Flank']
    new_indices = list(Region_idx_dict.values()) + [nonSpeRegionIDX] + [IDXPool]
    # print(list(Region_idx_dict.values()))
    # print(nonSpeRegionIDX)
    # print(IDXPool)
    for i, IDXtmp in enumerate(new_indices):
        ax.vlines(x=IDXtmp, ymin=2*FeatureLen, ymax=3 * FeatureLen, color=annotation_colors[i], label=SpeRegionLabel[i])
    LabelList = np.array([[i,color,label] for i, (color, label) in enumerate(zip(base_colors, baseName), start=1)], dtype=object)
    for i, (color, label) in enumerate(zip(base_colors, baseName), start=1):
        ax.vlines(x=i, ymin=PlotIdx-100.5, ymax=PlotIdx-99.5, color=color, label=label, zorder=3)
    for ClustI in np.unique(ClusterL):
        ax.vlines(x=ClusterDict[ClustI]+20, ymin=PlotIdx-200-0.5, ymax=PlotIdx-200+0.5, color=category_colors[ClusterDict[ClustI]], linewidth=3, zorder=3, label=ClustI)
    for SampleI in np.unique(SampleL):
        ax.vlines(x=Sample_dict[SampleI]+20, ymin=PlotIdx-200-0.5, ymax=PlotIdx-200+0.5, color=Sample_colors[Sample_dict[SampleI]], linewidth=3, zorder=3, label=SampleI)
    for CAI in np.unique(Rclust):
        ax.vlines(x=CAI+20, ymin=PlotIdx-200-0.5, ymax=PlotIdx-200+0.5, color=category_colors[-1*CAI-1], linewidth=3, zorder=3, label='Cluster %s' % CAI)
    readLabelLen = 0.05 * L 
    for labelIDX in np.arange(len(LabelYMIN)):
        LabelL = LabelYMIN[labelIDX]
        ClustI = ClusterL[labelIDX]
        CAI = CAtypeL[labelIDX]
        #print(labelIDX)
        SampleI = SampleL[labelIDX]       
        # print(SampleI)
        rect = patches.Rectangle((-readLabelLen, LabelL), readLabelLen-1, 1, linewidth=1, edgecolor=category_colors[ClusterDict[ClustI]], facecolor=category_colors[ClusterDict[ClustI]])
        ax.add_patch(rect)
        rect2 = patches.Rectangle((-2*readLabelLen, LabelL), readLabelLen-1, 1, linewidth=1, edgecolor=category_colors[-1*CAI-1], facecolor=category_colors[-1*CAI-1])
        ax.add_patch(rect2)
        rect3 = patches.Rectangle((-3*readLabelLen, LabelL), readLabelLen-1, 1, linewidth=1, edgecolor=Sample_colors[Sample_dict[SampleI]], facecolor=Sample_colors[Sample_dict[SampleI]])
        ax.add_patch(rect3)
    ax.set_xlim((-3*readLabelLen, L+1))
    ax.set_ylim((PlotIdx,3*FeatureLen))
    ax.set_yticks([])
    ax.set_xticks([])
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.1), ncol=5)
    ax.set_title(title)
    return(Rclust, readID)

def Main(sampleID, TDRecord, refFile,LabelList, saveDir, offset, mapQ, out_dir,out):
    # Visualize Data 
    TDchrom, TDStart, TDEnd = TDRecord.strip().split(",")[0],int(TDRecord.strip().split(",")[1]), int(TDRecord.strip().split(",")[2])
    ref = pysam.FastaFile(refFile)
    refSeq = ref.fetch(TDchrom, TDStart-offset, TDEnd+offset).upper()
    flank_5, flank_3 = ref.fetch(TDchrom, TDStart-offset, TDStart).upper(), ref.fetch(TDchrom, TDEnd, TDEnd+offset).upper()
    readTDSeq, readIDList= FetchTDsubSeq(TDRecord, saveDir)
    readTDSeqNew = readTDSeq                                                # sub read selection uncertain flank aln reads would be removed
    ReadIDs = readIDList
    seqdatamx_Raw = [refSeq] + readTDSeqNew
    readID = ['ref|Seq'] + ReadIDs
    consensus, msa = poa(seqdatamx_Raw,1)
    # Make Figure 
    fig = plt.figure(figsize=(12,6))
    # gs = GridSpec(1, 1, width_ratios=[1],height_ratios=[1])
    gs = GridSpec(1, 2, width_ratios=[1,1],height_ratios=[1])
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    # ax3 = fig.add_subplot(gs[1, 0])
    # ax4 = fig.add_subplot(gs[1, 1])
    # ax5 = fig.add_subplot(gs[2, 0])
    # ax6 = fig.add_subplot(gs[2, 1])
    SequencePlotRaw(ax1, seqdatamx_Raw, readID, 'RawData')
    # ax1 = fig.add_subplot(gs[0, 0])
    # SequencePlotCluster(seqdatamx_Raw, readID, flank_5, flank_3, hcutoff = 3, scutoff=0.05)
    Rclust, readID = SequencePlotCluster(ax2, seqdatamx_Raw, readID, flank_5, flank_3, out,'Clustering')
    plt.tight_layout()
    # plt.savefig(os.path.join(saveDir, '_'.join(TDRecord.strip().split("\t"))+".TRI.Visualize.pdf")) 
    # 导出为600dpi的PNG格式
    plt.savefig(os.path.join(saveDir, '_'.join(TDRecord.strip().split("\t"))+".Visualize.new.new.jpg"), dpi=600) 
    plt.close()

  
patientID = 'HCC10'
TSampleID = 'HCC10_N1,HCC10_N3,HCC10_N5,HCC10_T1,HCC10_T2,HCC10_T3,HCC10_T4,HCC10_T5,HCC10_T6,HCC10_T7'
NSampleID = 'HCC10_WBC'
# patientID = 'HCC9'
# TSampleID = 'HCC9_N1,HCC9_N3,HCC9_T1,HCC9_T2,HCC9_T3,HCC9_T4,HCC9_T5'
# NSampleID = 'HCC9_WBC'
# patientID = 'HCC8'
# TSampleID = 'HCC8_N1,HCC8_N3,HCC8_T1,HCC8_T2,HCC8_T3,HCC8_T4,HCC8_T5'
# NSampleID = 'HCC8_WBC'
# patientID = 'HCC13'
# TSampleID = 'HCC13_N1,HCC13_T1,HCC13_T2,HCC13_T3,HCC13_T4,HCC13_T5'
# NSampleID = 'HCC13_WBC'
TsampleID = TSampleID.split(",")
NsampleID = NSampleID.split(",")
Reference = '/NAS/wg_tkl/PanCancer_TKL/PanCancerRef/hg38_mainChr.fa'
savedir = '/NAS/wg_lyc/project/Evolution/dataPrepare/pancancerData/npz/offset50/Validation_npz.new/HCC10'
thread = 20
mapQ = 5
offset =50
LabelList = ["tumor" for x in TsampleID]  + ['normal' for x in NsampleID]


out_dir = savedir
windows = ['chr1_5387160_5387181','chr16_21908053_21908074','chr16_8737922_8737943' ,'chr20_29107403_29107424','chr4_49189404_49189425','chr6_72089929_72089950' ] # HCC10
# windows = ['chr10_133779204_133779225','chr21_9333096_9333117'] # HCC9
for i in windows:
    window = ','.join(i.split('_'))
    Main(patientID, window, Reference, LabelList, savedir,  offset, mapQ, out_dir, out)
    os.system('chmod 775 -R %s' % savedir)
