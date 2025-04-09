import os,re 
import numpy as np
import pandas as pd 
import pysam 
import gzip
from multiprocessing import Pool
from sklearn.cluster import DBSCAN
import argparse
import time
# env files 
faiFile = "/NAS/wg_tkl/PanCancer_TKL/PanCancerRef/hg38_mainChr.fa.fai"
eps=500
min_samples=3

def GetSpanReads(read_aln, INDELcutoff=40, CLIPcutoff=100):
    # input reads record from bed.gz file get every breakpoint 
    chrom, start, end, read_id, map_q, strand, cigarSTR = read_aln.strip().split("\t")
    uppercase_letters = np.array(re.findall(r'[A-Z]', cigarSTR))
    numbers = np.array([int(num) for num in re.findall(r'\d+', cigarSTR)])
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

def NonUniqReads(readIDXList):
    # remove non-unique aligned reads 
    Span = np.array([x.split("-") for x in readIDXList], dtype=int)
    SpanList = np.zeros(np.max(Span)+1)
    for S in Span:
        SpanList[np.arange(S[0], S[-1]+1)] += 1
    if np.where(SpanList>1)[0].shape[0] > 100:
        return('NonUnique-ALN')
    else:
        return('Uniq-ALN')

def SortReadSpan(readSpanList):
    # Get read span region 
    readSpanStart = np.array([x.split("-")[0] for x in readSpanList], dtype=int)
    return(np.argsort(readSpanStart))

def readsCLIP(CLIPRecord):
    # input CLIPRecord output candidate SV and SVType 
    chrom, refstart, readStart, readRegion, refRegion, strand, readID = list(CLIPRecord)
    IDXSort = SortReadSpan(readRegion)
    BPList = []
    if len(IDXSort) == 1:
        BPList.append(chrom[0] +":" + str(refstart[0]) +"_"+ chrom[0]+ ":" +str(refstart[0])+"|%s|SoloBP" % readID)
    else:
        i = 0
        while i < IDXSort.shape[0]-1:
            if readRegion[IDXSort[i]].split("-")[0] == str(readStart[IDXSort[i]]):
                BPList.append(chrom[IDXSort[i]] + ":" +str(refstart[IDXSort[i]]) +"_"+ chrom[IDXSort[i]] + ":" + str(refstart[IDXSort[i+1]]) +"|%s|SoloBP" % readID)
                i += 1
            else:     # readRegion[IDXSort[i]].split("-")[-1] == str(readStart[IDXSort[i]])   consider next CLIP Point distance to this point
                if (np.abs(readStart[IDXSort[i+1]] - readStart[IDXSort[i]])<50) and (readStart[IDXSort[i+1]] == int(readRegion[IDXSort[i+1]].split("-")[0])):
                    if (strand[IDXSort[i]] == strand[IDXSort[i+1]]) and (chrom[IDXSort[i]]==chrom[IDXSort[i+1]]) and (refstart[IDXSort[i+1]] < refstart[IDXSort[i]]) and (refstart[IDXSort[i+1]] > int(refRegion[IDXSort[i]].split(":")[-1].split("-")[0])):
                        BPList.append(chrom[IDXSort[i]] + ":" +str(refstart[IDXSort[i]]) + "_" + chrom[IDXSort[i]] + ":" +str(refstart[IDXSort[i+1]]) +"|%s|DUP" % readID)
                        if i+2 < IDXSort.shape[0]:
                            if readRegion[IDXSort[i+1]] == readRegion[IDXSort[i+2]]:
                                BPList.append(chrom[IDXSort[i]] + ":" +str(refstart[IDXSort[i]])+ "_"+ chrom[IDXSort[i]] +":"+str(refstart[IDXSort[i+1]]) +"|%s|DUP" % readID)
                                i += 3
                            else:
                                i += 2
                        else:
                            i += 2
                    elif (strand[IDXSort[i]] != strand[IDXSort[i+1]]) and (chrom[IDXSort[i]]==chrom[IDXSort[i+1]]):
                        BPList.append(chrom[IDXSort[i]] + ":" +str(refstart[IDXSort[i]]) + "_" + chrom[IDXSort[i]] +":"+str(refstart[IDXSort[i+1]]) +"|%s|INV" % readID)
                        i += 2
                    elif (chrom[IDXSort[i]]!=chrom[IDXSort[i+1]]):
                        BPList.append(chrom[IDXSort[i]] + ":" +str(refstart[IDXSort[i]]) + "_" + chrom[IDXSort[i+1]] +":"+str(refstart[IDXSort[i+1]]) +"|%s|TRA" % readID)
                        i += 2
                    else:
                        BPList.append(chrom[IDXSort[i]] + ":" +str(refstart[IDXSort[i]]) +"_"+ chrom[IDXSort[i]] + ":" + str(refstart[IDXSort[i]]) +"|%s|SoloBP" % readID)
                        i += 1
                else:
                    BPList.append(chrom[IDXSort[i]] + ":" +str(refstart[IDXSort[i]]) +"_"+ chrom[IDXSort[i]] + ":" + str(refstart[IDXSort[i]]) +"|%s|SoloBP" % readID)
                    i += 1
            if i >= IDXSort.shape[0]-1:
                break
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

def ParseWindows(bedFile, faiFile=faiFile, DataLabel='Tumor', cpu=60):
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
    ALNInfo = p.map(NonUniqReads, list(resDf_Group['ref_read']))
    p.close()
    del p
    resDf_Group['ALNInfo'] = ALNInfo
    resDf_Group_Uniq = resDf_Group.loc[resDf_Group['ALNInfo']=='Uniq-ALN']
    resDf_unique = resDf.loc[resDf['readID'].isin(resDf_Group_Uniq.index)]
    # Parse DEL and INS
    resDf_DEL = resDf.loc[(resDf['BPType']=='DEL')&(resDf['mapQ']>=5)]
    resDf_INS = resDf.loc[(resDf['BPType']=='INS')&(resDf['mapQ']>=5)]
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
    p = Pool(cpu)
    BPInfoList = p.map(readsCLIP, refDf_CLIP_Group_sub.to_numpy())
    p.close()
    del p
    resDf_CLIP_raw = pd.DataFrame([x.split("|") for x in np.concatenate(BPInfoList)], columns=['BPsite', 'readID', 'BPType'])
    resDf_CLIP_DUP = resDf_CLIP_raw.loc[resDf_CLIP_raw['BPType']=='DUP']
    resDf_CLIP_DUP['chrom'], resDf_CLIP_DUP['start'],resDf_CLIP_DUP['end'] = zip(*resDf_CLIP_DUP['BPsite'].apply(lambda x: RegionMaker(x)))
    ## Unspanned SVs 
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
    # Parse INV 
    resDf_CLIP_INV = resDf_CLIP_raw.loc[resDf_CLIP_raw['BPType']=='INV']
    resDf_CLIP_INV['Site'] = resDf_CLIP_INV['BPsite'].apply(lambda x: RegionEncoder(x, chromDict))
    resDf_CLIP_INV['DataLabel'] = DataLabel
    # Parse TRA 
    resDf_CLIP_TRA = resDf_CLIP_raw.loc[resDf_CLIP_raw['BPType']=='TRA']
    resDf_CLIP_TRA['Site'] = resDf_CLIP_TRA['BPsite'].apply(lambda x: RegionEncoder(x, chromDict))
    resDf_CLIP_TRA['DataLabel'] = DataLabel
    return(resDf_DEL, resDf_INS, resDf_CLIP_DUP, resDf_CLIP_INV, resDf_CLIP_TRA)

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

def RoughCompare(bedFileTumor, bedFileNormal, windowRecord, offset=50, cutoff=5):
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

def FindCandidateSVWindow(bedFileTumor, bedFileNormal, faiFile=faiFile, savedir="./", cpu=60):
    resDf_DEL_tumor, resDf_INS_tumor, resDf_CLIP_DUP_tumor, resDf_CLIP_INV_tumor, resDf_CLIP_TRA_tumor = ParseWindows(bedFileTumor, faiFile=faiFile, DataLabel='Tumor')
    resDf_DEL_normal, resDf_INS_normal, resDf_CLIP_DUP_normal, resDf_CLIP_INV_normal, resDf_CLIP_TRA_normal = ParseWindows(bedFileNormal, faiFile=faiFile, DataLabel='Normal')
    ## Double BreakPoint SVs 
    resDf_DEL_tumor.to_csv("%s/tmpDEL.bed" % savedir, sep="\t", header=None, index=False)
    os.system('sort -k1,1 -k2,2n %s/tmpDEL.bed -o %s/tmpDEL.bed && bedtools merge -i %s/tmpDEL.bed -d 200 -c 4,4 -o count_distinct,distinct | awk \'$4>3 {print $0"\tDEL"}\' > %s/CandidateDEL.tumor.merged.bed && rm %s/tmpDEL.bed' % (savedir, savedir, savedir, savedir, savedir))
    resDf_INS_tumor.to_csv("%s/tmpINS.bed" % savedir, sep="\t", header=None, index=False)
    os.system('sort -k1,1 -k2,2n %s/tmpINS.bed -o %s/tmpINS.bed && bedtools merge -i %s/tmpINS.bed -d 200 -c 4,4 -o count_distinct,distinct | awk \'$4>3 {print $0"\tINS"}\' > %s/CandidateINS.tumor.merged.bed && rm %s/tmpINS.bed' % (savedir, savedir, savedir, savedir, savedir))
    resDf_CLIP_DUP_tumor[['chrom', 'start', 'end', 'readID']].to_csv("%s/tmpDUP.bed" % savedir, sep="\t",header=None, index=False)
    os.system('sort -k1,1 -k2,2n %s/tmpDUP.bed -o %s/tmpDUP.bed && bedtools merge -i %s/tmpDUP.bed -d 200 -c 4,4 -o count_distinct,distinct | awk \'$4>3 {print $0"\tDUP"}\' > %s/CandidateDUP.tumor.merged.bed && rm %s/tmpDUP.bed' % (savedir, savedir, savedir, savedir, savedir))
    # Merge Double BreakPoint SVs 
    os.system('bedtools intersect -a %s/CandidateDUP.tumor.merged.bed -b %s/CandidateINS.tumor.merged.bed -wa -v > %s/CandidateSpan.tumor.merged.bed' % (savedir, savedir, savedir))
    os.system('bedtools intersect -a %s/CandidateDUP.tumor.merged.bed -b %s/CandidateINS.tumor.merged.bed -wa |sort -u >> %s/CandidateSpan.tumor.merged.bed' % (savedir, savedir, savedir))
    os.system('bedtools intersect -a %s/CandidateINS.tumor.merged.bed -b %s/CandidateDUP.tumor.merged.bed -wa -v >> %s/CandidateSpan.tumor.merged.bed' % (savedir, savedir, savedir))
    os.system('cat %s/CandidateDEL.tumor.merged.bed >> %s/CandidateSpan.tumor.merged.bed; sort -k1,1 -k2,2n %s/CandidateSpan.tumor.merged.bed -o %s/CandidateSpan.tumor.merged.bed' % (savedir, savedir, savedir, savedir))
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
    data = np.vstack(resDf_CLIP_INV['Site'])
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(data)
    labels = db.labels_
    resDf_CLIP_INV['labels'] = labels
    resDf_CLIP_INV_selected = resDf_CLIP_INV.loc[resDf_CLIP_INV['labels']!=-1]
    decisionDf = pd.DataFrame(resDf_CLIP_INV_selected.groupby(['labels'])['DataLabel'].apply(lambda x: np.array(x)))
    CandidateClusterINV = decisionDf.loc[(decisionDf['DataLabel'].apply(lambda x: np.where(x=='Tumor')[0].shape[0]) == decisionDf['DataLabel'].apply(len))&(decisionDf['DataLabel'].apply(len)>=3)].index
    CandidateINV = resDf_CLIP_INV_selected.loc[resDf_CLIP_INV_selected['labels'].isin(CandidateClusterINV)].sort_values(['labels'])
    CandidateINV.to_csv("%s/CandidateINV.tumor.merged.decision.bed" % savedir, sep="\t", index=False)
    # Get Candidate Window TRA 
    resDf_CLIP_TRA = pd.concat([resDf_CLIP_TRA_tumor, resDf_CLIP_TRA_normal], axis=0)
    data = np.vstack(resDf_CLIP_TRA['Site'])
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(data)
    labels = db.labels_
    resDf_CLIP_TRA['labels'] = labels
    resDf_CLIP_TRA_selected = resDf_CLIP_TRA.loc[resDf_CLIP_TRA['labels']!=-1]
    decisionDf = pd.DataFrame(resDf_CLIP_TRA_selected.groupby(['labels'])['DataLabel'].apply(lambda x: np.array(x)))
    CandidateClusterTRA = decisionDf.loc[(decisionDf['DataLabel'].apply(lambda x: np.where(x=='Tumor')[0].shape[0]) == decisionDf['DataLabel'].apply(len))&(decisionDf['DataLabel'].apply(len)>=3)].index
    CandidateTRA = resDf_CLIP_TRA_selected.loc[resDf_CLIP_TRA_selected['labels'].isin(CandidateClusterTRA)].sort_values(['labels'])
    CandidateTRA.to_csv("%s/CandidateTRA.tumor.merged.decision.bed" % savedir, sep="\t", index=False)
    return(["%s/CandidateSpan.tumor.merged.decision.bed" % savedir, "%s/CandidateINV.tumor.merged.decision.bed" % savedir, "%s/CandidateTRA.tumor.merged.decision.bed" % savedir])

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
    spanSV,INV,TRA = FindCandidateSVWindow(args.bedFileTumor, args.bedFileNormal, faiFile=args.faiFile, savedir=args.savedir, cpu=int(args.thread))
    # INV and TRA should directly write out into VCF format
    out_vcf = os.path.join(args.savedir, 'SomaticINV_BND.vcf')
    generate_vcfheaderINVTRA(args.faiFile, out_vcf, args.faiFile.split(".fai")[0])
    vcf = open(out_vcf, 'a')
    vcf.write(("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t{}\n").format(TumorID))
    # Check BND
    BNDDf = pd.read_csv('%s' % TRA, sep="\t")
    if BNDDf.shape[0] > 0:
        BNDDf['SortedBPList'] = BNDDf['BPsite'].apply(lambda x: SortBreakPoint(x, chromDict))
        BNDGroup = pd.concat([BNDDf.groupby(['labels'])['readID'].apply(lambda x: ",".join(list(np.unique(x)))), BNDDf.groupby(['labels'])['SortedBPList'].apply(lambda x: np.array(x))], axis=1)
        BNDGroup['BP1'], BNDGroup['BP2'] = zip(*BNDGroup['SortedBPList'].apply(lambda x: BPArrange(x)))
        for i in BNDGroup.index:
            BP1 = BNDGroup.loc[i, 'BP1']
            BP2 = BNDGroup.loc[i, 'BP2']
            ReadName = BNDGroup.loc[i, 'readID']
            ReadNum = str(len(ReadName.split(",")))
            vcf.write(BP1.split(":")[0] + "\t" + BP1.split(":")[1] + "\tTDScope.BND.%s-%s\t" % (BP1, BP2) + "N\tN]%s]\t.\tPASS\tSVLEN=-1;SVTYPE=BND;END=%s;SUPPORT=%s;RNAMES=%s\tGT\t0/1\n" % (BP2, BP2, ReadNum, ReadName))
            vcf.flush()
    # Check INV
    INVDf = pd.read_csv('%s' % INV, sep="\t")
    if INVDf.shape[0] > 0:
        INVDf['SortedBPList'] = INVDf['BPsite'].apply(lambda x: SortBreakPoint(x, chromDict))
        INVGroup = pd.concat([INVDf.groupby(['labels'])['readID'].apply(lambda x: ",".join(list(np.unique(x)))), INVDf.groupby(['labels'])['SortedBPList'].apply(lambda x: np.array(x))], axis=1)
        INVGroup['BP1'], INVGroup['BP2'] = zip(*INVGroup['SortedBPList'].apply(lambda x: BPArrange(x)))
        for i in INVGroup.index:
            BP1 = INVGroup.loc[i, 'BP1']
            BP2 = INVGroup.loc[i, 'BP2']
            SVLen = str(int(BP2.split(":")[-1]) - int(BP1.split(":")[-1]))
            ReadName = INVGroup.loc[i, 'readID']
            ReadNum = str(len(ReadName.split(",")))
            vcf.write(BP1.split(":")[0] + "\t" + BP1.split(":")[1] + "\tTDScope.INV.%s-%s\t" % (BP1, BP2) + "N\t<INV>\t.\tPASS\tSVLEN=%s;SVTYPE=BND;END=%s;SUPPORT=%s;RNAMES=%s\tGT\t0/1\n" % (SVLen, BP2.split(":")[-1], ReadNum, ReadName))
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


