'''
# Simple Reads Alignment Information
bedtools bamtobed -i <aim bamFile> -cigar | sort -k1,1 -k2,2n | bgzip > <bedFile>.bed.gz && tabix -p bed <bedFile>.bed.gz
# Updating record
## Version 3:
- adjust the GetSpanReads process, if strand == "-", calculate the reads alignment from 3' end;
- adjust CLIP SV choosing critiera, Alignment record must have mapping quality > mapQ threshold. the mapping quality is set to 5 bydefault. 
## Version 2: 
- Changed the logic of break point analysis, CLIP type breakpoint would be divided into SoloBP, INV, TRA, and Others; Where Others type including Duplication and Large Delation
- Chnaged the way we find Large Duplication / Delation, through breakpoints clustering methods 
- Chnaged the name of output vcf from SomaticINV_BND.vcf to InterALNSVs.vcf
- Add Large duplication and Delation result to InterALNSVs.vcf
- Adjust the overlap of large del and inner alignment del with bedtools intersect -f 0.5 -r parameters
- Adjust the overlap of large dup and inner alignment ins with bedtools intersect -f 0.5 -r parameters to avoid unlimited merge of candidate window 
- Final output should be InterALNSVs.vcf + CandidateSpan.tumor.merged.decision.somatic.bed, the later should go through TDScope process for further analysis.

'''
import os,re 
import numpy as np
import pandas as pd 
import pysam 
import gzip
from multiprocessing import Pool
from sklearn.cluster import DBSCAN
import argparse
import time
import functools

# env files 
faiFile = "/NAS/wg_tkl/PanCancer_TKL/PanCancerRef/hg38_mainChr.fa.fai"
eps=500
min_samples=3

def GetSpanReads(read_aln, INDELcutoff=40, CLIPcutoff=100):
    # input reads record from bed.gz file get every breakpoint 
    chrom, start, end, read_id, map_q, strand, cigarSTR = read_aln.strip().split("\t")
    uppercase_letters = np.array(re.findall(r'[A-Z]', cigarSTR))
    numbers = np.array([int(num) for num in re.findall(r'\d+', cigarSTR)])
    if strand == "-":
        uppercase_letters = uppercase_letters[::-1]
        numbers = numbers[::-1]
    MatchIDX = np.where(np.in1d(uppercase_letters, ['M','X']))[0]
    readsGrowthIDX = np.where(np.in1d(uppercase_letters, ['H','S','I']))[0]
    refGrowthIDX = np.where(np.in1d(uppercase_letters, ['D', 'P', 'N']))[0]
    DELIDX = np.where(np.in1d(uppercase_letters, ['D']))[0]
    INSIDX = np.where(np.in1d(uppercase_letters, ['I']))[0]
    CLIPIDX = np.where(np.in1d(uppercase_letters, ['S','H']))[0]
    readStart, readEnd = np.sum(numbers[0:MatchIDX[0]]), np.sum(numbers[np.setdiff1d(np.arange(MatchIDX[-1]+1), refGrowthIDX)])
    BPList = []
    # Parse DEL
    for i in DELIDX:
        if numbers[i] >= INDELcutoff:
            refstart = int(start) + numbers[MatchIDX[np.where(MatchIDX<i)]].sum() + numbers[refGrowthIDX[np.where(refGrowthIDX<i)]].sum()
            refend = refstart + numbers[i]
            readstart = numbers[MatchIDX[np.where(MatchIDX<i)]].sum() + numbers[refGrowthIDX[np.where(refGrowthIDX<i)]].sum()
            readend = readstart + 0
            BPList.append([chrom, refstart, refend, read_id, readstart,readend, chrom+":"+start+"-"+end, str(readStart)+"-"+str(readEnd), int(map_q), strand, 'DEL'])
    for i in INSIDX:
        if numbers[i] >= INDELcutoff:
            refstart = int(start) + numbers[MatchIDX[np.where(MatchIDX<i)]].sum() + numbers[refGrowthIDX[np.where(refGrowthIDX<i)]].sum()
            refend = refstart + 0
            readstart = numbers[MatchIDX[np.where(MatchIDX<i)]].sum() + numbers[refGrowthIDX[np.where(refGrowthIDX<i)]].sum()
            readend = readstart + numbers[i]
            BPList.append([chrom, refstart, refend, read_id, readstart, readend, chrom+":"+start+"-"+end, str(readStart)+"-"+str(readEnd), int(map_q), strand, 'INS'])
    for i in CLIPIDX:
        if numbers[i] >= CLIPcutoff:
            refstart = int(start) + numbers[MatchIDX[np.where(MatchIDX<i)]].sum() + numbers[refGrowthIDX[np.where(refGrowthIDX<i)]].sum()
            refend = refstart + 0
            if i == 0:
                readstart = readStart
                readend = readstart
            else:
                readstart = readEnd
                readend = readEnd
            BPList.append([chrom, refstart, refend, read_id, readstart, readend, chrom+":"+start+"-"+end, str(readStart)+"-"+str(readEnd), int(map_q), strand, 'CLIP'])
    return(BPList)

def NonUniqReads(readIDXList, cutoff=100):
    # remove non-unique aligned reads 
    Span = np.array([x.split("-") for x in readIDXList], dtype=int)
    SpanList = np.zeros(np.max(Span)+1)
    for S in Span:
        SpanList[np.arange(S[0], S[-1]+1)] += 1
    if np.where(SpanList>1)[0].shape[0] > cutoff:
        return('NonUnique-ALN')
    else:
        return('Uniq-ALN')

def NonUniqDetail(readIDXList):
    # check status of reads non unique alignment status  
    Span = np.array([x.split("-") for x in np.unique(readIDXList)], dtype=int)
    SpanList = np.zeros(np.max(Span)+1)
    for S in Span:
        SpanList[np.arange(S[0], S[-1]+1)] += 1
    SpanListSub = SpanList[np.min(Span):]
    return(np.where(SpanListSub>1)[0].shape[0] / SpanListSub.shape[0])

def SortReadSpan(readSpanList):
    # Get read span region 
    readSpanStart = np.array([x.split("-")[0] for x in readSpanList], dtype=int)
    return(np.argsort(readSpanStart))

def readsCLIP2(CLIPRecord, mergecutoff=100):
    # Merge CLIP Point on read located within the mergecutoff threshold 
    chrom, refstart, readStart, readRegion, refRegion, strand, readID = list(CLIPRecord)
    splitSite = np.where(np.diff(readStart)>mergecutoff)[0]
    GroupIDX = np.split(np.arange(len(readStart)), splitSite+1)
    BPList = []
    for G in GroupIDX:
        # Check NonUniq aligned status 
        OverLapRate = NonUniqDetail(readRegion[G])
        if OverLapRate < 0.5:                      # should at least lower than half, then check the breakpoint's status 
            if G.shape[0] == 1:
                BPList.append(chrom[G[0]] +":" + str(refstart[G[0]]) +"_"+ chrom[G[0]]+ ":" +str(refstart[G[0]])+"|%s|SoloBP" % readID)
            else:
                if np.unique(readRegion[G]).shape[0] == 1:
                    for g in G:
                        BPList.append(chrom[g] +":" + str(refstart[g]) +"_"+ chrom[g]+ ":" +str(refstart[g])+"|%s|SoloBP" % readID)
                else:
                    IDXSort = SortReadSpan(readRegion[G])
                    r = 0
                    while r < len(IDXSort)-1:
                        currentRegion = readRegion[G[IDXSort[r]]]
                        nextRegion = readRegion[G[IDXSort[r+1]]]
                        currentRef = chrom[G[IDXSort[r]]] +":" + str(refstart[G[IDXSort[r]]])
                        nextRef = chrom[G[IDXSort[r+1]]] +":" + str(refstart[G[IDXSort[r+1]]])
                        currentchrom, nextchrom = chrom[G[IDXSort[r]]], chrom[G[IDXSort[r+1]]]
                        currentstrand = strand[G[IDXSort[r]]]
                        nextstrand = strand[G[IDXSort[r+1]]]
                        if currentRegion == nextRegion:     # current CLIP point is SoloBP
                            BPList.append(currentRef +"_"+ currentRef +"|%s|SoloBP" % readID)
                            r += 1
                        else:     # Could be DUP, INV, TRA, Largescale DEL 
                            if currentchrom != nextchrom:
                                BPList.append(currentRef +"_"+ nextRef +"|%s|TRA" % readID)
                            elif currentstrand != nextstrand:
                                BPList.append(currentRef +"_"+ nextRef +"|%s|INV" % readID)
                            else:
                                BPList.append(currentRef +"_"+ nextRef +"|%s|Others" % readID)
                            r += 2
                        if r == (len(IDXSort)-1):
                            BPList.append(currentRef +"_"+ currentRef +"|%s|SoloBP" % readID)
    return(BPList)

def RegionEncoder(BPInfo, chromDict):
    chrom1,BP1,chrom2,BP2 = re.split(r'[:_]', BPInfo)
    BP1, BP2 = int(BP1), int(BP2)
    Point1, Point2 = BP1 + chromDict[chrom1], BP2 + chromDict[chrom2]
    if Point1 <=Point2:
        return(np.array([Point1,Point2]))
    else:
        return(np.array([Point2, Point1]))

def RegionMaker(BPInfo):
    # make region for DUP type SV chrom1, chrom2 should be the same 
    chrom1,BP1,chrom2,BP2 = re.split(r'[:_]', BPInfo)
    BP1, BP2 = int(BP1), int(BP2)
    if chrom1 == chrom2:
        if BP1 <= BP2:
            return(chrom1, str(BP1), str(BP2))
        else:
            return(chrom1,str(BP2),str(BP1))

def SortBreakPoint(BPInfo, chromDict):
    # sort BreakPoint, output [chromA:LociA, chromB:LociB]
    chrom1,BP1,chrom2,BP2 = re.split(r'[:_]', BPInfo)
    BP1, BP2 = int(BP1), int(BP2)
    Point1, Point2 = BP1 + chromDict[chrom1], BP2 + chromDict[chrom2]
    if Point1 <=Point2:
        return(np.array([chrom1+":"+str(BP1),chrom2+":"+str(BP2)]))
    else:
        return(np.array([chrom2+":"+str(BP2),chrom1+":"+str(BP1)]))

def BPArrange(SortBPList):
    # sort two BP to get representation sites 
    BPList = np.vstack(SortBPList)
    BP1List = BPList[:,0]
    BP2List = BPList[:,1]
    chrom1 = BP1List[0].split(":")[0]
    BP1 = str(int(np.mean([int(x.split(":")[-1]) for x in BP1List])))
    chrom2 = BP2List[0].split(":")[0]
    BP2 = str(int(np.mean([int(x.split(":")[-1]) for x in BP2List])))
    return(chrom1 +":"+BP1, chrom2+":"+BP2)

def faiToChromDict(faiFile):
    # parse faiFile to get chromosome encoding data
    LenList = []
    chrom = np.array(['chr%s' % x for x in range(1,23)] + ['chrX', 'chrY', 'chrM'])
    with open(faiFile) as input:
        for records in input.readlines():
            c, Len = records.split("\t")[0:2]
            LenList.append(int(Len))
    LenList = np.array(LenList, dtype=int)
    chromDict = {}
    for i,C in enumerate(chrom):
        chromDict[C] = np.sum(LenList[:i])
    return(chromDict)

def ParseWindows(bedFile, faiFile=faiFile, DataLabel='Tumor', cpu=60, mapQ=5):
    # parse bed.gz file 
    p = Pool(cpu)
    with gzip.open(bedFile, 'rt') as bed:
        results = p.map(GetSpanReads, bed.readlines())
    p.close()
    del p
    rescollect = [] 
    for r in results:
        if len(r) != 0:
            rescollect += r
    resDf = pd.DataFrame(rescollect)
    resDf.columns = ['chrom', 'refStart', 'refEnd', 'readID', 'readStart', 'readEnd', 'refRegion', 'readRegion', 'mapQ', 'strand', 'BPType']
    resDf['ref_read'] = resDf['refRegion'] +"|"+resDf['readRegion']
    resDf_Group = pd.DataFrame(resDf.groupby(['readID'])['ref_read'].apply(lambda x: [x.split("|")[-1] for x in np.unique(x)]))
    resDf_Group['readID'] = resDf_Group.index
    # remove Non-UniqALN reads
    p = Pool(cpu)
    NonUniqReads_exe = functools.partial(NonUniqReads, cutoff=100)
    ALNInfo = p.map(NonUniqReads_exe, list(resDf_Group['ref_read']))
    p.close()
    del p
    resDf_Group['ALNInfo'] = ALNInfo
    resDf_Group_Uniq = resDf_Group.loc[resDf_Group['ALNInfo']=='Uniq-ALN']
    resDf_unique = resDf.loc[resDf['readID'].isin(resDf_Group_Uniq.index)]
    # resDf_unique = resDf
    # Parse DEL and INS inner-alignment breakpoints 
    resDf_DEL = resDf.loc[(resDf['BPType']=='DEL')&(resDf['mapQ']>=mapQ)]
    resDf_INS = resDf.loc[(resDf['BPType']=='INS')&(resDf['mapQ']>=mapQ)]
    # Parse DUP
    refDf_CLIP = resDf_unique.loc[(resDf_unique['BPType']=='CLIP')].sort_values(['readID', 'readStart'])
    refDf_CLIP_Group = pd.concat([refDf_CLIP.groupby(['readID'])['chrom'].apply(lambda x: np.array(x)), 
                                    refDf_CLIP.groupby(['readID'])['refStart'].apply(lambda x: np.array(x)), 
                                    refDf_CLIP.groupby(['readID'])['readStart'].apply(lambda x: np.array(x)),
                                    refDf_CLIP.groupby(['readID'])['readRegion'].apply(lambda x: np.array(x)),
                                    refDf_CLIP.groupby(['readID'])['refRegion'].apply(lambda x: np.array(x)),
                                    refDf_CLIP.groupby(['readID'])['strand'].apply(lambda x: np.array(x))], axis=1)
    refDf_CLIP_Group_sub = refDf_CLIP_Group.loc[refDf_CLIP_Group['readRegion'].apply(len)>1]
    refDf_CLIP_Group_sub['readID'] = refDf_CLIP_Group_sub.index
    # Label each inter alignment type breakpoints
    p = Pool(cpu)
    BPInfoList = p.map(readsCLIP2, refDf_CLIP_Group_sub.to_numpy())
    p.close()
    del p
    resDf_CLIP_raw = pd.DataFrame([x.split("|") for x in np.concatenate(BPInfoList)], columns=['BPsite', 'readID', 'BPType'])
    ## Unspanned SVs 
    chromDict = faiToChromDict(faiFile)
    # Parse inter-alignment breakpoint within same chromosome 
    P = Pool(cpu)
    RegionEncoder_exe = functools.partial(RegionEncoder, chromDict=chromDict)
    SitePool = P.map(RegionEncoder_exe, list(resDf_CLIP_raw['BPsite']))
    P.close()
    del P
    resDf_CLIP_raw['Site'] = SitePool
    resDf_CLIP_raw['DataLabel'] = DataLabel
    # Solo BPs 
    resDf_CLIP_Others = resDf_CLIP_raw.loc[resDf_CLIP_raw['BPType']=='Others']
    # Parse INV 
    resDf_CLIP_INV = resDf_CLIP_raw.loc[resDf_CLIP_raw['BPType']=='INV']
    # Parse TRA 
    resDf_CLIP_TRA = resDf_CLIP_raw.loc[resDf_CLIP_raw['BPType']=='TRA']
    return(resDf_DEL, resDf_INS, resDf_CLIP_Others, resDf_CLIP_INV, resDf_CLIP_TRA, refDf_CLIP_Group_sub)

def FetchAimRegion(read_aln, refstart,refend):
    # fetch span reads length within aim reference region
    chrom, start, end, read_id, map_q, strand, cigarSTR = read_aln.strip().split("\t")
    uppercase_letters = np.array(re.findall(r'[A-Z]', cigarSTR))
    numbers = np.array([int(num) for num in re.findall(r'\d+', cigarSTR)])
    refGrowth = np.array(['D', 'P', 'N', 'M','X'])
    readGrowth = np.array(['H','S','I', 'M', 'X'])
    MatchIDX = np.where(np.in1d(uppercase_letters, ['M','X']))[0]
    readsGrowthIDX = np.where(np.in1d(uppercase_letters, ['H','S','I']))[0]
    refGrowthIDX = np.where(np.in1d(uppercase_letters, ['D', 'P', 'N']))[0]
    DELIDX = np.where(np.in1d(uppercase_letters, ['D']))[0]
    INSIDX = np.where(np.in1d(uppercase_letters, ['I']))[0]
    CLIPIDX = np.where(np.in1d(uppercase_letters, ['S','H']))[0]
    readStart, readEnd = np.sum(numbers[0:MatchIDX[0]]), np.sum(numbers[np.setdiff1d(np.arange(MatchIDX[-1]+1), refGrowthIDX)])
    # Make reflect matrix 
    refLoci,readLoci = [int(start)],[0]
    tmprefStart, tmpreadStart = int(start), 0
    for i,C in enumerate(uppercase_letters):
        if C in refGrowth:
            tmprefStart += numbers[i]
        refLoci.append(tmprefStart)
        if C in readGrowth:
            tmpreadStart += numbers[i]
        readLoci.append(tmpreadStart)
        # Find Break Site on reads 
    Site_5, Site_3 = np.nan, np.nan
    if int(start) < refstart:
        tmp_5 = np.where(np.array(refLoci)<=refstart)[0][-1]
        offset5 = refstart - refLoci[tmp_5]
        Site_5 = readLoci[tmp_5] + offset5
    else:
        Site_5 = readStart
    if int(end) > refend:
        tmp_3 = np.where(np.array(refLoci)<=refend)[0][-1]
        offset3 = refend - refLoci[tmp_3]
        Site_3 = readLoci[tmp_3] + offset3
    else:
        Site_3 = readEnd
    return([read_id, int(start), int(end), Site_5, Site_3])

def RoughCompare(bedFileTumor, bedFileNormal, windowRecord, offset=40, cutoff=5):
    # Designed for double breakpoint somatic SV windows 
    chrom,start,end = windowRecord.strip().split("\t")[0:3]
    start,end = int(start), int(end)
    windowType = windowRecord.strip().split("\t")[-1]
    T = pysam.TabixFile(bedFileTumor)
    N = pysam.TabixFile(bedFileNormal)
    T_reads = pd.DataFrame([FetchAimRegion(x,start,end) for x in T.fetch(chrom,start,end) if (int(x.split("\t")[4])>=cutoff)], columns=['readID', 'refstart', 'refend', 'readstart', 'readend'])
    N_reads = pd.DataFrame([FetchAimRegion(x,start,end) for x in N.fetch(chrom,start,end)], columns=['readID', 'refstart', 'refend', 'readstart', 'readend'])
    TDf = pd.concat([T_reads.groupby(['readID'])['refstart'].apply(lambda x: np.array(x)), 
                        T_reads.groupby(['readID'])['refend'].apply(lambda x: np.array(x)),
                        T_reads.groupby(['readID'])['readstart'].apply(lambda x: np.array(x)),
                        T_reads.groupby(['readID'])['readend'].apply(lambda x: np.array(x))], axis=1)
    NDf = pd.concat([N_reads.groupby(['readID'])['refstart'].apply(lambda x: np.array(x)), 
                        N_reads.groupby(['readID'])['refend'].apply(lambda x: np.array(x)),
                        N_reads.groupby(['readID'])['readstart'].apply(lambda x: np.array(x)),
                        N_reads.groupby(['readID'])['readend'].apply(lambda x: np.array(x))], axis=1)
    # Filter SpanReads 
    TDf_span = TDf.loc[(TDf['refstart'].apply(lambda x:np.min(x))<=start)&(TDf['refend'].apply(lambda x: np.max(x))>=end)]
    NDf_span = NDf.loc[(NDf['refstart'].apply(lambda x:np.min(x))<=start)&(NDf['refend'].apply(lambda x: np.max(x))>=end)]
    TDf_span['Length'] = TDf_span['readend'].apply(np.max) - TDf_span['readstart'].apply(np.min)
    NDf_span['Length'] = NDf_span['readend'].apply(np.max) - NDf_span['readstart'].apply(np.min)
    # Decide whether window would be candidate somatic 
    if windowType in ['DEL']: # DEL type SV 
        tmpDf = TDf_span.loc[TDf_span['Length']<np.min(NDf_span['Length'])-offset]
    else:
        tmpDf = TDf_span.loc[TDf_span['Length']>np.max(NDf_span['Length'])+offset]
    if tmpDf.shape[0] >= 3:
        return("{chrom}\t{start}\t{end}\t{TumorSpan}\t{NormalSpan}\t{windowType}\tCandidateSom".format(
            chrom=chrom, start=start,end=end, TumorSpan=TDf_span.shape[0], NormalSpan=NDf_span.shape[0], windowType=windowType
        ))
    else:
        return("{chrom}\t{start}\t{end}\t{TumorSpan}\t{NormalSpan}\t{windowType}\tGermlineWindow".format(
            chrom=chrom, start=start,end=end, TumorSpan=TDf_span.shape[0], NormalSpan=NDf_span.shape[0], windowType=windowType
        ))

def DecisionWithDBSCAN(resDf_aim, eps=500, min_samples=3, AimLabel='Tumor'):
    # DBSCAN clustering with tumor and normal breakpoint, select tumor only cluster as the candidate somatic 
    # return decision Df
    data = np.vstack(resDf_aim['Site'])
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(data)
    labels = db.labels_
    resDf_aim['labels'] = labels
    resDf_CLIP_INV_selected = resDf_aim.loc[resDf_aim['labels']!=-1]
    decisionDf = pd.DataFrame(resDf_CLIP_INV_selected.groupby(['labels'])['DataLabel'].apply(lambda x: np.array(x)))
    CandidateClusterINV = decisionDf.loc[(decisionDf['DataLabel'].apply(lambda x: np.where(x==AimLabel)[0].shape[0]) == decisionDf['DataLabel'].apply(len))&(decisionDf['DataLabel'].apply(len)>=min_samples)].index
    return(resDf_aim.loc[resDf_aim['labels'].isin(CandidateClusterINV)].sort_values('labels'))

def FindCandidateSVWindow(bedFileTumor, bedFileNormal, faiFile=faiFile, savedir="./", cpu=60):
    resDf_DEL_tumor, resDf_INS_tumor, resDf_CLIP_Others_tumor, resDf_CLIP_INV_tumor, resDf_CLIP_TRA_tumor, CLIP_Raw_tumor = ParseWindows(bedFileTumor, faiFile=faiFile, DataLabel='Tumor')
    resDf_DEL_normal, resDf_INS_normal, resDf_CLIP_Others_normal, resDf_CLIP_INV_normal, resDf_CLIP_TRA_normal, CLIP_Raw_normal = ParseWindows(bedFileNormal, faiFile=faiFile, DataLabel='Normal', mapQ=0)
    chromDict = faiToChromDict(faiFile)
    ## Double BreakPoint SVs 
    resDf_DEL_tumor.to_csv("%s/tmpDEL.bed" % savedir, sep="\t", header=None, index=False)
    os.system('sort -k1,1 -k2,2n %s/tmpDEL.bed -o %s/tmpDEL.bed && bedtools merge -i %s/tmpDEL.bed -d 200 -c 4,4 -o count_distinct,distinct | awk \'$4>3 {print $0"\tDEL"}\' > %s/CandidateDEL.tumor.merged.bed && rm %s/tmpDEL.bed' % (savedir, savedir, savedir, savedir, savedir))
    resDf_INS_tumor.to_csv("%s/tmpINS.bed" % savedir, sep="\t", header=None, index=False)
    os.system('sort -k1,1 -k2,2n %s/tmpINS.bed -o %s/tmpINS.bed && bedtools merge -i %s/tmpINS.bed -d 200 -c 4,4 -o count_distinct,distinct | awk \'$4>3 {print $0"\tINS"}\' > %s/CandidateINS.tumor.merged.bed && rm %s/tmpINS.bed' % (savedir, savedir, savedir, savedir, savedir))
    # Treat the potential Dup / DEL type SV 
    resDf_CLIP_Others = pd.concat([resDf_CLIP_Others_tumor, resDf_CLIP_Others_normal])
    Candidate_CLIP_Others = DecisionWithDBSCAN(resDf_CLIP_Others)
    # Label DUP and DEL 
    Candidate_CLIP_Others['SVType'] = Candidate_CLIP_Others['BPsite'].apply(lambda x: "DEL" if int(x.split("_")[0].split(":")[-1]) < int(x.split("_")[-1].split(":")[-1]) else "DUP")
    # Filter should have at least 3 support reads and same SVType 
    Candidate_CLIP_Others_Stat = pd.concat([Candidate_CLIP_Others.groupby(['labels'])['SVType'].apply(lambda x: np.unique(x)),
                                            Candidate_CLIP_Others.groupby(['labels'])['readID'].apply(lambda x: np.unique(x))], axis=1).reset_index()
    SelectGroups = list(Candidate_CLIP_Others_Stat.loc[(Candidate_CLIP_Others_Stat['SVType'].apply(len)==1)&(Candidate_CLIP_Others_Stat['readID'].apply(len)>=3), 'labels'])
    Candidate_CLIP_Others_filter = Candidate_CLIP_Others.loc[Candidate_CLIP_Others['labels'].isin(SelectGroups)]
    Candidate_CLIP_Others_filter['SortedBPList'] = Candidate_CLIP_Others_filter['BPsite'].apply(lambda x: SortBreakPoint(x, chromDict))
    Candidate_CLIP_Others_Result = pd.concat([Candidate_CLIP_Others_filter.groupby(['labels'])['SortedBPList'].apply(lambda x: [a.split(":")[0] for a in np.vstack(x)[:,0]][0]), 
                                                Candidate_CLIP_Others_filter.groupby(['labels'])['SortedBPList'].apply(lambda x: np.min([int(a.split(":")[-1]) for a in np.vstack(x)[:,0]])),
                                                Candidate_CLIP_Others_filter.groupby(['labels'])['SortedBPList'].apply(lambda x: np.min([int(a.split(":")[-1]) for a in np.vstack(x)[:,1]])), 
                                                Candidate_CLIP_Others_filter.groupby(['labels'])['SVType'].apply(lambda x: list(x)[0]), 
                                                Candidate_CLIP_Others_filter.groupby(['labels'])['readID'].apply(lambda x: ",".join(list(np.unique(x))))], axis=1)
    Candidate_CLIP_Others_Result.columns = ['chrom', 'start', 'end', 'SVType','readID']
    # Treat DEL type SV first 
    ## For Large Scale DEL output into decision bed file 
    Candidate_CLIP_Others_Result.loc[Candidate_CLIP_Others_Result['SVType']=='DEL'].to_csv('%s/CandidateLargeDEL.tumor.merged.bed' % savedir, sep="\t", header=None, index=False)
    os.system('sort -k1,1 -k2,2n %s/CandidateLargeDEL.tumor.merged.bed -o %s/CandidateLargeDEL.tumor.merged.bed' % (savedir, savedir))
    os.system('bedtools intersect -a %s/CandidateLargeDEL.tumor.merged.bed -b %s/CandidateDEL.tumor.merged.bed -wa -v -f 0.5 -r > %s/CandidateLargeDEL.tumor.merged.decision.bed' % (savedir, savedir, savedir))
    dfDEL_large = pd.read_csv("%s/CandidateLargeDEL.tumor.merged.decision.bed" % savedir, sep="\t",header=None)
    Candidate_CLIP_Others_Result_DEL_IDX = Candidate_CLIP_Others_Result.loc[Candidate_CLIP_Others_Result['readID'].isin(list(dfDEL_large[4]))].index
    Candidate_CLIP_Others_Result_DEL = Candidate_CLIP_Others_filter.loc[Candidate_CLIP_Others_filter['labels'].isin(Candidate_CLIP_Others_Result_DEL_IDX)]
    Candidate_CLIP_Others_Result_DEL['BPType'] = 'DEL'
    Candidate_CLIP_Others_Result_DEL[['BPsite','readID','BPType','Site','DataLabel','labels']].to_csv("%s/CandidateLargeDEL.tumor.merged.decision.bed" % savedir, sep="\t", index=False)
    # For intersect DEL get new SVborder, prepare for further reads gain loss test 
    os.system('bedtools intersect -a %s/CandidateLargeDEL.tumor.merged.bed -b %s/CandidateDEL.tumor.merged.bed -wa -wb -f 0.5 -r > %s/CandidateLargeDEL.vs.CandidateDEL.intersect.bed' % (savedir, savedir, savedir))
    dfDEL = pd.read_csv("%s/CandidateLargeDEL.vs.CandidateDEL.intersect.bed" % savedir, sep="\t",header=None)
    dfDEL['start'] = dfDEL.apply(lambda x:x[1] if x[1]<=x[6] else x[6], axis=1)
    dfDEL['end'] = dfDEL.apply(lambda x:x[2] if x[1]>=x[7] else x[7], axis=1)
    dfDEL['supportReads'] = dfDEL.apply(lambda x: ",".join(list(np.unique(x[4].split(",") + x[9].split(",")))), axis=1)
    dfDEL['supReadsCount'] = dfDEL['supportReads'].apply(lambda x:len(x.split(",")))
    dfDEL[[0,'start','end','supReadsCount', 'supportReads', 3]].to_csv("%s/CandidateIntersect.DEL.tumor.merged.bed" % savedir, sep="\t", header=None, index=False)
    # Treat DUP type SV next
    ## For Large Scale DUP output into decision bed file 
    Candidate_CLIP_Others_Result.loc[Candidate_CLIP_Others_Result['SVType']=='DUP'].to_csv('%s/CandidateDUP.tumor.merged.bed' % savedir, sep="\t", header=None, index=False)
    os.system('sort -k1,1 -k2,2n %s/CandidateDUP.tumor.merged.bed -o %s/CandidateDUP.tumor.merged.bed' % (savedir, savedir))
    os.system('bedtools intersect -a %s/CandidateDUP.tumor.merged.bed -b %s/CandidateINS.tumor.merged.bed -wa -v -f 0.5 -r > %s/CandidateDUP.tumor.merged.decision.bed' % (savedir, savedir, savedir))
    dfDUP_large = pd.read_csv("%s/CandidateDUP.tumor.merged.decision.bed" % savedir, sep="\t",header=None)
    Candidate_CLIP_Others_Result_DUP_IDX = Candidate_CLIP_Others_Result.loc[Candidate_CLIP_Others_Result['readID'].isin(list(dfDUP_large[4]))].index
    Candidate_CLIP_Others_Result_DUP = Candidate_CLIP_Others_filter.loc[Candidate_CLIP_Others_filter['labels'].isin(Candidate_CLIP_Others_Result_DUP_IDX)]
    Candidate_CLIP_Others_Result_DUP['BPType'] = 'DUP'
    Candidate_CLIP_Others_Result_DUP[['BPsite','readID','BPType','Site','DataLabel','labels']].to_csv("%s/CandidateDUP.tumor.merged.decision.bed" % savedir, sep="\t", index=False)
    # For intersect DUP get new SVborder, prepare for further reads gain loss test 
    os.system('bedtools intersect -a %s/CandidateDUP.tumor.merged.bed -b %s/CandidateINS.tumor.merged.bed -wa -wb -f 0.5 -r > %s/CandidateDUP.vs.CandidateINS.intersect.bed' % (savedir, savedir, savedir))
    dfDUP = pd.read_csv("%s/CandidateDUP.vs.CandidateINS.intersect.bed" % savedir, sep="\t",header=None)
    dfDUP['start'] = dfDUP.apply(lambda x:x[1] if x[1]<=x[6] else x[6], axis=1)
    dfDUP['end'] = dfDUP.apply(lambda x:x[2] if x[1]>=x[7] else x[7], axis=1)
    dfDUP['supportReads'] = dfDUP.apply(lambda x: ",".join(list(np.unique(x[4].split(",") + x[9].split(",")))), axis=1)
    dfDUP['supReadsCount'] = dfDUP['supportReads'].apply(lambda x:len(x.split(",")))
    dfDUP[[0,'start','end','supReadsCount', 'supportReads', 3]].to_csv("%s/CandidateIntersect.DUP.tumor.merged.bed" % savedir, sep="\t", header=None, index=False)
    # Merge Double BreakPoint SVs 
    os.system('cat %s/CandidateIntersect.DUP.tumor.merged.bed > %s/CandidateSpan.tumor.merged.bed' % (savedir, savedir))
    os.system('cat %s/CandidateIntersect.DEL.tumor.merged.bed >> %s/CandidateSpan.tumor.merged.bed' % (savedir, savedir))
    os.system('bedtools intersect -a %s/CandidateINS.tumor.merged.bed -b %s/CandidateIntersect.DUP.tumor.merged.bed -wa -v >> %s/CandidateSpan.tumor.merged.bed' % (savedir, savedir, savedir))
    os.system('bedtools intersect -a %s/CandidateDEL.tumor.merged.bed -b %s/CandidateIntersect.DEL.tumor.merged.bed -wa -v >> %s/CandidateSpan.tumor.merged.bed' % (savedir, savedir, savedir))
    # Filter candidate somatic SV regions 
    P = Pool(cpu)
    results = []
    with open("%s/CandidateSpan.tumor.merged.bed" % savedir) as input:
        for windowRecord in input.readlines():
            async_result = P.apply_async(RoughCompare, (bedFileTumor, bedFileNormal, windowRecord, ))
            results.append(async_result)
    with open("%s/CandidateSpan.tumor.merged.decision.bed" % savedir, 'w') as f:
        f.write("chrom\tstart\tend\tTumorSpan\tNormalSpan\twindowType\twindowLabel\n")
        while results:
            for result in results:
                if result.ready():
                    output = result.get()
                    f.write(output + '\n')
                    f.flush()
                    results.remove(result)
    P.terminate()
    os.system('grep CandidateSom %s/CandidateSpan.tumor.merged.decision.bed | awk \'{print $1"\t"$2"\t"$3"\t"$4"\t"$5"\t"$6}\' > %s/CandidateSpan.tumor.merged.decision.somatic.bed' % (savedir, savedir))
    ## Single BreakPoint SVs 
    # Get Candidate Window INV 
    resDf_CLIP_INV = pd.concat([resDf_CLIP_INV_tumor, resDf_CLIP_INV_normal], axis=0)
    CandidateINV = DecisionWithDBSCAN(resDf_CLIP_INV)
    CandidateINV[['BPsite','readID','BPType','Site','DataLabel','labels']].to_csv("%s/CandidateINV.tumor.merged.decision.bed" % savedir, sep="\t", index=False)
    # Get Candidate Window TRA 
    resDf_CLIP_TRA = pd.concat([resDf_CLIP_TRA_tumor, resDf_CLIP_TRA_normal], axis=0)
    CandidateTRA = DecisionWithDBSCAN(resDf_CLIP_TRA)
    CandidateTRA[['BPsite','readID','BPType','Site','DataLabel','labels']].to_csv("%s/CandidateTRA.tumor.merged.decision.bed" % savedir, sep="\t", index=False)
    return(["%s/CandidateSpan.tumor.merged.decision.bed" % savedir,"%s/CandidateLargeDEL.tumor.merged.decision.bed" % savedir, "%s/CandidateDUP.tumor.merged.decision.bed" % savedir,  "%s/CandidateINV.tumor.merged.decision.bed" % savedir, "%s/CandidateTRA.tumor.merged.decision.bed" % savedir])

def generate_vcfheaderINVTRA(faiFile,out_vcf,fasta):
    chromosomes = {}
    with open(faiFile) as input:
        for records in input.readlines():
            chrom,length = records.strip().split("\t")[0:2]
            chromosomes[chrom] = int(length)
    Info ='''##INFO=<ID=SVTYPE,Number=1,Type=String,Description="Type of structural variant">\n##INFO=<ID=SVLEN,Number=1,Type=Integer,Description="Length of the SV">\n##INFO=<ID=END,Number=1,Type=Integer,Description="End position of the SV">\n##INFO=<ID=SUPPORT,Number=1,Type=Integer,Description="Number of reads supporting the structural variation">\n##INFO=<ID=RNAMES,Number=.,Type=String,Description="Names of supporting reads">\n##INFO=<ID=AF,Number=1,Type=Float,Description="Allele Frequency">\n'''
    Tools = '''##fileformat=VCFv4.2\n##source=TDscope.1.0\n##FILTER=<ID=PASS,Description="All filters passed">\n'''
    with open(out_vcf,'w') as vcf:
        ### Tools 
        vcf.write(Tools)
        ### Data
        current_time = time.strftime("%Y/%m/%d %H:%M:%S", time.localtime())
        vcf.write('''##fileDate="'''+current_time+'''"\n''')
        ### reference info
        vcf.write('''##reference='''+fasta+'\n')
        for chrom,length in chromosomes.items():
            vcf.write('''##contig=<ID='''+chrom+',length='+str(length)+'>\n')
        ### SV info
        vcf.write('''##ALT=<ID=LargeDEL,Description="Large Delation">\n##ALT=<ID=LargeDUP,Description="Large Duplication">\n''')
        vcf.write('''##ALT=<ID=INV,Description="Invasion">\n##ALT=<ID=BND,Description="Translocation">\n''')
        ### format
        vcf.write('''##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n''')
        ### INFO
        vcf.write(Info)
    return(out_vcf)

def main(args):
    TumorID = os.path.basename(args.bedFileTumor).split(".bed")[0]
    if not os.path.exists(args.savedir):
        os.system('mkdir %s' % args.savedir)
    # Load ChromLength
    LenList = []
    chrom = np.array(['chr%s' % x for x in range(1,23)] + ['chrX', 'chrY', 'chrM'])
    with open(faiFile) as input:
        for records in input.readlines():
            c, Len = records.split("\t")[0:2]
            LenList.append(int(Len))
    LenList = np.array(LenList, dtype=int)
    chromDict = {}
    for i,C in enumerate(chrom):
        chromDict[C] = np.sum(LenList[:i])
    spanSV,DEL,DUP,INV,TRA = FindCandidateSVWindow(args.bedFileTumor, args.bedFileNormal, faiFile=args.faiFile, savedir=args.savedir, cpu=int(args.thread))
    # INV and TRA should directly write out into VCF format
    out_vcf = os.path.join(args.savedir, 'InterALNSVs.vcf')
    generate_vcfheaderINVTRA(args.faiFile, out_vcf, args.faiFile.split(".fai")[0])
    vcf = open(out_vcf, 'a')
    vcf.write(("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t{}\n").format(TumorID))
    # Check BND
    BNDDf = pd.read_csv('%s' % TRA, sep="\t")
    if BNDDf.shape[0] > 0:
        BNDDf['SortedBPList'] = BNDDf['BPsite'].apply(lambda x: SortBreakPoint(x, chromDict))
        BNDGroup = pd.concat([BNDDf.groupby(['labels'])['readID'].apply(lambda x: ",".join(list(np.unique(x)))), 
                              BNDDf.groupby(['labels'])['SortedBPList'].apply(lambda x: np.array(x))], axis=1)
        BNDGroup['BP1'], BNDGroup['BP2'] = zip(*BNDGroup['SortedBPList'].apply(lambda x: BPArrange(x)))
        for i in BNDGroup.index:
            BP1 = BNDGroup.loc[i, 'BP1']
            BP2 = BNDGroup.loc[i, 'BP2']
            ReadName = BNDGroup.loc[i, 'readID']
            ReadNum = str(len(ReadName.split(",")))
            vcf.write(BP1.split(":")[0] + "\t" + BP1.split(":")[1] + "\tTDScope.BND.%s-%s_1\t" % (BP1, BP2) +
                      "N\tN]%s]\t.\tPASS\tSVLEN=-1;SVTYPE=BND;MATE_ID=TDScope.BND.%s-%s_2;SUPPORT=%s;RNAMES=%s\tGT\t0/1\n" % 
                      (BP2, BP1, BP2, ReadNum, ReadName))
            vcf.write(BP2.split(":")[0] + "\t" + BP2.split(":")[1] + "\tTDScope.BND.%s-%s_2\t" % (BP1, BP2) +
                      "N\tN]%s]\t.\tPASS\tSVLEN=-1;SVTYPE=BND;MATE_ID=TDScope.BND.%s-%s_1;SUPPORT=%s;RNAMES=%s\tGT\t0/1\n" % 
                      (BP1, BP1, BP2, ReadNum, ReadName))
            vcf.flush()
    # Check INV
    INVDf = pd.read_csv('%s' % INV, sep="\t")
    if INVDf.shape[0] > 0:
        INVDf['SortedBPList'] = INVDf['BPsite'].apply(lambda x: SortBreakPoint(x, chromDict))
        INVGroup = pd.concat([INVDf.groupby(['labels'])['readID'].apply(lambda x: ",".join(list(np.unique(x)))), 
                              INVDf.groupby(['labels'])['SortedBPList'].apply(lambda x: np.array(x))], axis=1)
        INVGroup['BP1'], INVGroup['BP2'] = zip(*INVGroup['SortedBPList'].apply(lambda x: BPArrange(x)))
        for i in INVGroup.index:
            BP1 = INVGroup.loc[i, 'BP1']
            BP2 = INVGroup.loc[i, 'BP2']
            SVLen = str(int(BP2.split(":")[-1]) - int(BP1.split(":")[-1]))
            ReadName = INVGroup.loc[i, 'readID']
            ReadNum = str(len(ReadName.split(",")))
            vcf.write(BP1.split(":")[0] + "\t" + BP1.split(":")[1] + "\tTDScope.INV.%s-%s\t" % (BP1, BP2) + "N\t<INV>\t.\tPASS\tSVLEN=%s;SVTYPE=INV;END=%s;SUPPORT=%s;RNAMES=%s\tGT\t0/1\n" % (SVLen, BP2.split(":")[-1], ReadNum, ReadName))
            vcf.flush()
    # Check Large DUP / DEL 
    DELDf = pd.read_csv('%s' % DEL, sep="\t")
    if DELDf.shape[0] > 0:
        DELDf['SortedBPList'] = DELDf['BPsite'].apply(lambda x: SortBreakPoint(x, chromDict))
        DELGroup = pd.concat([DELDf.groupby(['labels'])['readID'].apply(lambda x: ",".join(list(np.unique(x)))), 
                              DELDf.groupby(['labels'])['SortedBPList'].apply(lambda x: np.array(x))], axis=1)
        DELGroup['BP1'], DELGroup['BP2'] = zip(*DELGroup['SortedBPList'].apply(lambda x: BPArrange(x)))
        for i in DELGroup.index:
            BP1 = DELGroup.loc[i, 'BP1']
            BP2 = DELGroup.loc[i, 'BP2']
            SVLen = str(int(BP2.split(":")[-1]) - int(BP1.split(":")[-1]))
            ReadName = DELGroup.loc[i, 'readID']
            ReadNum = str(len(ReadName.split(",")))
            vcf.write(BP1.split(":")[0] + "\t" + BP1.split(":")[1] + "\tTDScope.DEL.%s-%s\t" % (BP1, BP2) + "N\t<LargeDEL>\t.\tPASS\tSVLEN=-%s;SVTYPE=LargeDEL;END=%s;SUPPORT=%s;RNAMES=%s\tGT\t0/1\n" % (SVLen, BP2.split(":")[-1], ReadNum, ReadName))
            vcf.flush()
    DUPDf = pd.read_csv('%s' % DUP, sep="\t")
    if DUPDf.shape[0] > 0:
        DUPDf['SortedBPList'] = DUPDf['BPsite'].apply(lambda x: SortBreakPoint(x, chromDict))
        DUPGroup = pd.concat([DUPDf.groupby(['labels'])['readID'].apply(lambda x: ",".join(list(np.unique(x)))), 
                              DUPDf.groupby(['labels'])['SortedBPList'].apply(lambda x: np.array(x))], axis=1)
        DUPGroup['BP1'], DUPGroup['BP2'] = zip(*DUPGroup['SortedBPList'].apply(lambda x: BPArrange(x)))
        for i in DUPGroup.index:
            BP1 = DUPGroup.loc[i, 'BP1']
            BP2 = DUPGroup.loc[i, 'BP2']
            SVLen = str(int(BP2.split(":")[-1]) - int(BP1.split(":")[-1]))
            ReadName = DUPGroup.loc[i, 'readID']
            ReadNum = str(len(ReadName.split(",")))
            vcf.write(BP1.split(":")[0] + "\t" + BP1.split(":")[1] + "\tTDScope.DUP.%s-%s\t" % (BP1, BP2) + "N\t<LargeDUP>\t.\tPASS\tSVLEN=-%s;SVTYPE=LargeDUP;END=%s;SUPPORT=%s;RNAMES=%s\tGT\t0/1\n" % (SVLen, BP2.split(":")[-1], ReadNum, ReadName))
            vcf.flush()
    vcf.close()
    return(spanSV)

if __name__ =='__main__':
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("-T", "--bedFileTumor", required=True, help="Tumor Reads Alignment bed file")
    parser.add_argument("-N", "--bedFileNormal", required=True, help="Normal Reads Alignment bed file")
    parser.add_argument("-f", "--faiFile", required=True, help="faiFile path for reference genome")
    parser.add_argument("-s", "--savedir", required=True, help="Output path")
    parser.add_argument("-t", "--thread", required=True, help="CPU use for program")
    args = parser.parse_args()
    main(args)

