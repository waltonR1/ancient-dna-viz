#######################################################################################################################################
##                    genetics library to read and manipulate packedancestrymap data in native python                                ##
#######################################################################################################################################
# 
# author : Pr. Jérémie SUBLIME, jsublime@isep.fr
# version : january 2024
#
# This Python library proposes some basic functions to handle packedancestrymap data (such as the ones from the Allen Ancient DNA Resource) without the need for any 3rd party software or R language. The proposed library relies on numpy and pandas and is as such makes genetics data easy to process and visualize using suited Machine Learning or Deep Learning methods readily available in Python.
#
# 该 Python 库提供一系列基础函数，用于在纯 Python 环境下读取与操作 PackedAncestryMap 格式的基因组数据
# （例如 Allen Ancient DNA Resource 数据集），无需依赖 R 或其他第三方专用软件。
# 库依赖 numpy 与 pandas，可以方便地在 Python 中直接使用机器学习或深度学习方法处理与可视化遗传数据。
#
# The file contains basic functions to load the data from .geno, .anno, .ind and .snp files, and turn them into suitable numpy or pandas data matrices. Given the large amount of missing values in these data, we also provide custom versions of the PseudoHamming distance and PseudoHamming ratio, as well as wrapped imputation methods from sklearn.
#
# 本文件包含从 .geno、.anno、.ind 和 .snp 文件加载数据的函数，
# 可将其转换为适合分析的 numpy 数组或 pandas 数据框格式。
# 鉴于这些基因组数据中存在大量缺失值，文件中还提供了自定义的伪哈明距离（PseudoHamming distance）
# 和伪哈明比率（PseudoHamming ratio）计算方法，并封装了 sklearn 的缺失值填补方法。
#
# The first part of the file contains only native Python functions, some of which have a pre-compilation option to speed them up (they can be identified by the @jit(nopython=True) decorator). All these functions can be called from native Python or from a Jupyter Notebook.
#
# 文件的第一部分仅包含纯 Python 实现的函数，其中部分函数可通过 @jit(nopython=True) 装饰器实现预编译以加速。
# 所有这些函数都可以直接在 Python 脚本或 Jupyter Notebook 中调用。
#
# The second part of the file contains CUDA functions that can be run on GPU to speed up computations on large data. These functions will run only if you have a proper CUDA set up ! Furthermore, while functions with the "_cuda" suffix are the CUDA wrappers for functions equivalents to the ones from the first part of the file and can be called from native Python or from a Jupyter Notebook, functions with the "_cudakernel" suffix are CUDA kernels and should not be called from plain Python code unless you have some basic CUDA knowledge. We also advise caution if you decide to modify the functions with a "_cudakernel" suffix : they look like Python code, but they are parallelized CUDA code.
#
# 文件的第二部分包含可在 GPU 上运行的 CUDA 函数，用于加速大规模数据运算。
# 这些函数只有在正确安装 CUDA 环境后才能运行。
# 名称带有 “_cuda” 后缀的函数是前一部分函数的 GPU 封装版本，可从 Python 或 Notebook 调用；
# 而带有 “_cudakernel” 后缀的函数是 CUDA 内核函数，不应直接在普通 Python 代码中调用，
# 除非用户具备基本的 CUDA 编程知识。
# 修改这些函数时需谨慎，因为它们虽然看似 Python 语法，但实质是并行执行的 CUDA 代码。
#
#
import pandas as pd
from numba import jit
import math
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra


##### loadRawGenoFile(filename,ext=False)
#
#   Load and prepare a .geno file and extract some key characteristics
#
#   parameters:
#       filename : path/filename
#       ext : if set to "False", the ".geno" extension is added
#
#   returns:
#       geno_file : the raw file encoded
#       nind : the number of lines/individuals in the file
#       nsnp : the number of columns/snps per individual
#       rlen : the record length of each row
#
def loadRawGenoFile(filename,ext=False):
    """
    读取并准备 .geno 文件，提取基本特征信息。
        - 打开文件并读取头部信息；
        - 提取样本数与 SNP 数；
        - 计算每行的记录长度。

    :param filename: 文件路径（可不带“.geno”扩展名）
    :param ext: 是否已包含“.geno”扩展名，默认为 False
    :return: (geno_file, nind, nsnp, rlen)
             - geno_file: 打开的二进制文件对象
             - nind: 个体数量（样本数）
             - nsnp: SNP 数量
             - rlen: 每行记录长度（字节数）
    """
    if(ext):
        geno_file=open(filename, 'rb')
    else:
        geno_file=open(filename+".geno", 'rb')

    header=geno_file.read(20)
    nind,nsnp=[int(x) for x in header.split()[1:3]]
    rlen=max(48,int(np.ceil(nind*2/8)))
    geno_file.seek(rlen)
    return geno_file,nind,nsnp,rlen



##### unpackfullgenofile(filename)
#
#   Unpack a .geno file
#
#   parameters:
#       filename : path/filename
#
#   returns:
#       geno : the geno file as a numpy array
#       nind : the number of lines/individuals in the file
#       nsnp : the number of columns/snps per individual
#       rlen : the record length of each row
#
def unpackfullgenofile(filename):
    """
    解包完整的 .geno 文件，将其转换为 numpy 数组。
        - 调用 loadRawGenoFile() 获取结构信息；
        - 解码原始字节并展开为显式的 0/1 序列；
        - 返回完整的基因型矩阵。

    :param filename: .geno 文件路径
    :return: (geno, nind, nsnp, rlen)
             - geno: 解包后的 numpy 数组
             - nind: 个体数量
             - nsnp: SNP 数量
             - rlen: 每行记录长度
    """
    geno_file, nind,nsnp,rlen=loadRawGenoFile(filename,True)
    geno=np.fromfile(filename, dtype='uint8')[rlen:]
    geno.shape=(nsnp,rlen)
    geno=np.unpackbits(geno,axis=1)[:,:(2*nind)]
    return geno,nind,nsnp,rlen



##### unpackAndFilterSNPs(genofilename,snpIndexes)
#
#   Unpack a geno data and pre-filter the SNPs
#
#   parameters:
#       geno : raw numpy geno file
#       snpIndexes : an index list of SNPs to keep /!\ must be extracted from a compatible SNP file with the same indexing !
#       nind : number of individuals
#
#   returns:
#       geno : the geno file as a numpy array
#
def unpackAndFilterSNPs(geno,snpIndexes,nind):
    """
    解包并筛选指定 SNP 索引的基因型数据。
        - 使用 numpy.unpackbits() 解压；
        - 仅保留目标 SNP；
        - 将两位编码转换为基因型数值。

    :param geno: 原始 numpy 编码基因型矩阵
    :param snpIndexes: 要保留的 SNP 索引列表（与 .snp 文件对应）
    :param nind: 个体数量
    :return: 过滤并解码后的 SNP 数组
    """
    geno=np.unpackbits(geno,axis=1)[snpIndexes,:(2*nind)]
    geno=2*geno[:,::2]+geno[:,1::2]
    return geno


##### genofileToCSV(filename,delim=",")
#
#   Convert a .geno file into a large CSV. /!\ heavy memory and disk load !
#
#   parameters:
#       filename : path/filename
#       delim    : the column separator for the csv file
#
#   returns: nothing. Create a csv file in the same folder as the original .geno file.
#
def genofileToCSV(filename,delim=";"):
    """
    将 .geno 文件转换为 CSV 格式。
        - 解包所有 SNP；
        - 将 0/1 二进制转化为 0/1/2 型基因矩阵；
        - 保存为 CSV 文件。

    :param filename: .geno 文件路径
    :param delim: CSV 列分隔符，默认 ";"
    :return: 无返回值（在原路径下生成 .csv 文件）
    """
    geno, nind,nsnp,rlen=unpackfullgenofile(filename)
    geno=2*geno[:,::2]+geno[:,1::2]
    np.savetxt(filename+".csv", geno.astype(int),fmt="%i", delimiter=delim)


##### genofileToPandas(filename,snpfilename,indfilename,transpose=True)
#
#    Convert a .geno file into a pandas dataframe. /!\ heavy memory and disk load !
#
#   parameters:
#       filename : path/filename for the .geno
#       snpfilename : path/filename for the .snp
#       indfilename : path/filename for the .ind
#       transpose : boolean parameter to transpose the final matrix or not
#
#   returns:
#       df : the created pandas dataframe
#
def genofileToPandas(filename,snpfilename,indfilename,transpose=True):
    """
    将 .geno 文件转换为 pandas DataFrame。
        - 解包所有 SNP；
        - 读取对应的 .snp 与 .ind 文件；
        - 合并并构建以样本与 SNP 为索引的矩阵。

    :param filename: .geno 文件路径
    :param snpfilename: .snp 文件路径
    :param indfilename: .ind 文件路径
    :param transpose: 是否转置矩阵（默认 True）
    :return: 转换后的 pandas DataFrame
    """
    geno,nind,nsnp,rlen=unpackfullgenofile(filename)
    geno=2*geno[:,::2]+geno[:,1::2]
    ind=pd.read_csv(indfilename, sep=r"\s+", header=None)
    snp=pd.read_csv(snpfilename, sep=r"\s+", header=None)
    df=pd.DataFrame(geno,columns=ind[0],index=snp[0])

    #df=df.replace(to_replace=[3], value=[""],regex=True) #missing values
    if(transpose):
        df=df.T

    return df


##### CreateLocalityFile(annofilename,sep="	",toCSV=False,minSNPnbr=-1)
#
#    Browse through the .anno file to figure out the world region to wich the indiviual belong and to filter duplicate individuals
#
#   parameters:
#       annofilename : path and name of the .anno file
#       sep : separator used in the .anno file
#       toCSV : boolean, if set to True, will save the resulting table as a csv file
#       verbose : display the different steps of the process
#       minSNPnbr : minimal number of SNPs covered to consider the individual
#       hapl : If set to True, the haplogroups (when known) will be included in the resulting matrix
#
#   returns:
#       df : the resulting pandas dataframe
#
def CreateLocalityFile(annofilename,sep="	",toCSV=False,verbose=False,minSNPnbr=-1,hapl=False):
    """
    从 .anno 文件中提取个体地理信息并去除重复项。
        - 统计每个个体的 SNP 覆盖度；
        - 去重并仅保留覆盖度最高的记录；
        - 根据国家/地区映射所属“World Zone”；
        - 可选输出为 CSV 文件。

    :param annofilename: .anno 文件路径
    :param sep: 文件分隔符，默认制表符
    :param toCSV: 是否导出为 CSV 文件
    :param verbose: 是否输出处理进度信息
    :param minSNPnbr: 最小 SNP 覆盖阈值，小于此值的个体将被过滤
    :param hapl: 是否包含 Y/mtDNA 单倍群信息
    :return: 含地理映射信息的 pandas DataFrame
    """
    if(verbose):
        print("Loading anno file")
    anno=pd.read_csv(annofilename, sep=sep,low_memory=False)

    if(verbose):
        print("Mapping SNPs coverage")
    anno['coverage']=anno['SNPs hit on autosomal targets (Computed using easystats on 1240k snpset)'].where(anno['SNPs hit on autosomal targets (Computed using easystats on 1240k snpset)']!="..",other=anno['SNPs hit on autosomal targets (Computed using easystats on HO snpset)'])
    anno['coverage']=anno['coverage'].astype(int)

    if(verbose):
        print("Filtering duplicate individuals, and keeping only the ones with the highest number of SNPs")

    dp_indexes=anno.duplicated(subset=['Master ID'], keep=False)
    anno['duplicate']=dp_indexes
    for i in range(0,anno.shape[0]):
        if(anno['duplicate'].iloc[i]):
            subset=anno[anno["Master ID"]==anno['Master ID'].iloc[i]]
            max_coverage=max(anno['coverage'])
            anno.loc[((anno["Master ID"]==anno['Master ID'].iloc[i]) & (anno["coverage"]==max_coverage)),'duplicate']=False
    anno=anno.drop(anno[anno['duplicate'] == True].index)
    if(minSNPnbr>0):
        anno=anno.drop(anno[anno['coverage'] < minSNPnbr].index)

    df=anno[["Genetic ID"]].copy()
    df["Political Entity"]=anno["Political Entity"]
    df["Locality"]=anno["Locality"]
    df["Molecular Sex"]=anno["Molecular Sex"]
    df["Master ID"]=anno["Master ID"]
    df["World Zone"]="Unknown"
    if(hapl):
        df["Y haplogroup"]=anno['Y haplogroup (manual curation in ISOGG format)']
        df["mt haplogroup"]=anno['mtDNA haplogroup if >2x or published']

    if(verbose):
        print(str(df.shape[0])+" individuals were kept")
        print("Mapping World Regions")

    #Might be questionnable : feel free to move countries around ...
    Weurope=["Switzerland","Netherlands","France","United Kingdom","Gernamy","Germany","Belgium","Channel Islands","Ireland","Isle of Man","Luxembourg"]
    Neurope=["Finland","Iceland","Norway","Greenland","Denmark","Faroes","Sweden"]
    Ceurope=["Slovakia","Hungary","Czech","Poland","Austria","Slovenia"]
    Seurope=["Italy","Greece","Turkey","Spain","Malta","Albania","Cyprus","Gibraltar","North Macedonia","Portugal"]
    #We might have some issues with Eastern Russia locations
    Eeurope=["Serbia","Latvia","Crimea","Bosnia-Herzegovina","Montenegro","Romania","Estonia","Belarus","Abkhazia","Lithuania","Russia","Armenia","Moldova","Georgia","Ukraine","Croatia","Bulgaria"]
    Nafrica=["Algeria","Morocco","Libya","Tunisia","Canary Islands"]
    Wafrica=["Senegal","Nigeria","Gambia","Sierra Leone"]
    Eafrica=["Ethiopia","Sudan","Kenya","Eritrea"]
    CSafrica=["Zambia","Uganda","Cameroon","Malawi","South Africa","Congo","Central African Republic","Namibia","Lesotho","Botswana","Angola","Tanzania"]
    middleeast=["Iran","Israel","Syria","Egypt","Lebanon","Saudi Arabia","Jordan","Yemen","Iraq"]
    Sasia=["Pakistan","India","Bangladesh","Afghanistan","Sri Lanka","Nepal"]
    Casia=["Azerbaijan","Uzbekistan","Kazakhstan","Tajikistan","Kyrgyzstan","Turkmenistan"]
    Easia=["Mongolia","China","Taiwan","Korea","Japan"]
    SEasia=["Cambodia","Vietnam","Thailand","Myanmar","Indonesia","Brunei","Philippines","Singapore","Laos","Malaysia"]
    oceania=["Vanuatu","Tonga","Solomon Islands","New Zealand","Papua New Guinea","Australia","Federated States of Micronesia","Polynesia","Guam"]
    Camerica=["St. Lucia","Puerto Rico","Panama","Haiti","Guadeloupe","Mexico","Cuba","Bahamas","Belize","Dominican Republic","Barbados","Curacao"]
    Namerica=["Canada","USA"]
    Samerica=["Uruguay","Venezuela","Colombia","Brazil","Peru","Bolivia","Argentina","Chile"]
    others=["..","HumanReferenceSequence"]

    for c in Weurope:
        df["World Zone"][df["Political Entity"].str.contains(c, na=False)]="Europe (W)"
    for c in Neurope:
        df["World Zone"][df["Political Entity"].str.contains(c, na=False)]="Europe (N)"
    for c in Ceurope:
        df["World Zone"][df["Political Entity"].str.contains(c, na=False)]="Europe (C)"
    for c in Seurope:
        df["World Zone"][df["Political Entity"].str.contains(c, na=False)]="Europe (S)"
    for c in Eeurope:
        df["World Zone"][df["Political Entity"].str.contains(c, na=False)]="Europe (E)"
    for c in Nafrica:
        df["World Zone"][df["Political Entity"].str.contains(c, na=False)]="Africa (N)"
    for c in Eafrica:
        df["World Zone"][df["Political Entity"].str.contains(c, na=False)]="Africa (E)"
    for c in Wafrica:
        df["World Zone"][df["Political Entity"].str.contains(c, na=False)]="Africa (W)"
    for c in CSafrica:
        df["World Zone"][df["Political Entity"].str.contains(c, na=False)]="Africa (SC)"
    for c in middleeast:
        df["World Zone"][df["Political Entity"].str.contains(c, na=False)]="Middle East"
    for c in Sasia:
        df["World Zone"][df["Political Entity"].str.contains(c, na=False)]="Asia (S)"
    for c in Casia:
        df["World Zone"][df["Political Entity"].str.contains(c, na=False)]="Asia (C)"
    for c in SEasia:
        df["World Zone"][df["Political Entity"].str.contains(c, na=False)]="Asia (SE)"
    for c in Easia:
        df["World Zone"][df["Political Entity"].str.contains(c, na=False)]="Asia (E)"
    for c in oceania:
        df["World Zone"][df["Political Entity"].str.contains(c, na=False)]="Oceania"
    for c in Camerica:
        df["World Zone"][df["Political Entity"].str.contains(c, na=False)]="America (C)"
    for c in Samerica:
        df["World Zone"][df["Political Entity"].str.contains(c, na=False)]="America (S)"
    for c in Namerica:
        df["World Zone"][df["Political Entity"].str.contains(c, na=False)]="America (N)"

    #Russian case
    ERussia=["Primorsky","Novaya","Ayon","Yanranay","Naukan","Khabarovsk","Zabaykalsky","Buryat","Irkutsk","Tuva","Nelemnoe","Kolymskoe","Rytkuchi","Sireniki","Bogorodskoe","Bulava","Nizhniy Gavan","Djigda","New Chaplino","Voyampolka","Nikolskoye","Chokyrdah","Kovran","Sakha","Kamchatski","Bakhta","Eastern Sayan","Siberia"]
    CRussia=["Lagoda","Altai","Udmurtia","Dagestan","Bashkiria","Khakassia","Omsk","Astrakhan","Kemerovo","Tyumen","Tatarstan","Artybash and Kebezen","Chechen","Kalmykia","Okrug","Chuvash","Konda","Sos'va","Dudinka","Ust'-Avam","Volochanka","Potapovo","Farkovo","Turukhansk","Kellog","Sulomai","Baklanikha","Surgutikha","Pakulikha","Igarka","Samara"]
    for c in CRussia:
        df["World Zone"][anno["Locality"].str.contains(c, na=False)]="Asia (C)"
    for c in ERussia:
        df["World Zone"][anno["Locality"].str.contains(c, na=False)]="Asia (E)"

    if(toCSV):
        if(verbose):
            print("Exporting location CSV")
        df.to_csv(annofilename+"_loc.csv",sep=';')

    if(verbose):
        print("Done with the region mapping")

    return df



##### unpack22chrDNAwithLocations(genofilename,snpfilename, annofilename,chro=None,transpose=True,toCSV=False,verbose=False)
#
#    Extract the first 22 chromosome's DNA for all individuals while removing duplicates and extracting locations
#    Remarks :
#       - The location is based on the CreateLocalityFile() function, same for the filtering
#       - Will be slow and will require a lot of memory !
#       - We suggest running this only once, exporting the CSV, and then working from the CSV files
#
#   parameters:
#       filename : path/filename for the .geno
#       snpfilename : path/filename for the .snp
#       annofilename : path/filename for the .anno
#       chro : an array (empty by default) to choose to extract only some chromosomes, e.g.: [1,2,21]
#       transpose : boolean parameter to transpose the final matrix or not
#       toCSV : boolean parameter, if set to true CSV files with the resulting matrices will be created in the same folder as the geno file
#       to_numpy : if set to True, will skip the creation of the pandas DataFrame and return a numpy array instead (will save memory and time)
#       verbose : Will discribe the current advancement if set to True
#       minSNPnbr : minimal number of SNPs covered to consider the individual, if 0<minSNPnbr<=1 a percentage will be considered
#       hardhaplfilter : Will remove all individual whose haplogroup is unknown if set to True and if Chromosome 24 is included.
#
#
#   returns:
#       df : the data loaded as a Pandas Dataframe or Numpy array (depending on the to_numpy parameter)
#       annowithloc : the matching location dataframe for all individuals
#
def unpack22chrDNAwithLocations(genofilename,snpfilename, annofilename,chro=None,transpose=True,toCSV=False,to_numpy=True,verbose=False,minSNPnbr=-1,hardhaplfilter=False):
    """
    解包前 22 条常染色体数据并附带个体地理信息。
        - 从 .snp 文件中过滤指定染色体；
        - 解包相应 SNP 基因型；
        - 通过 .anno 文件去重、获取地理信息；
        - 可选择仅保留 Y 染色体（Chromosome 24）个体；
        - 支持导出 CSV。

    :param genofilename: .geno 文件路径
    :param snpfilename: .snp 文件路径
    :param annofilename: .anno 文件路径
    :param chro: 要提取的染色体列表（如 [1,2,21]），默认前 22 条
    :param transpose: 是否转置输出矩阵
    :param toCSV: 是否导出结果 CSV 文件
    :param to_numpy: 是否返回 numpy 数组（节省内存）
    :param verbose: 是否打印进度
    :param minSNPnbr: 最小 SNP 覆盖度阈值，支持比例形式 (0<val<=1)
    :param hardhaplfilter: 若为 True 且含 Y 染色体，则过滤未知单倍群个体
    :return: (df, annowithloc)
             - df: DNA 基因型数据 (DataFrame 或 numpy)
             - annowithloc: 含地理信息的注释表
    """
    if(verbose):
        print("Opening SNP file")
    snp=pd.read_csv(snpfilename, sep=r"\s+", header=None)
    if(verbose):
        print("Filtering the chromosomes indexes")

    ychrom=False

    if(chro!=None):
        chr22dna=[]
        for ch in chro:
            filterId=snp[snp[1]==ch].index.tolist()
            if(len(filterId)>0):
                chr22dna.extend(filterId)
            if(ch==24):
                ychrom=True
    else:
        chr22dna=snp[snp[1] < 23].index.tolist()

    if(verbose):
        print( "Unpacking DNA /!\ This will take a while and a lot of memory /!\ ")
    geno_file, nind,nsnp,rlen=loadRawGenoFile(genofilename,True)
    geno=np.fromfile(genofilename, dtype='uint8')[rlen:]
    geno.shape=(nsnp,rlen)
    geno=unpackAndFilterSNPs(geno,chr22dna,nind)

    if(ychrom==True):
        annowithloc=CreateLocalityFile(annofilename,verbose=verbose,toCSV=toCSV,hapl=True)
        if(hardhaplfilter):
            if(verbose):
                print( "Filtering empty Y haplogroups")
            excludefilters=["na"," ",".."]
            malesId=annowithloc[annowithloc['Molecular Sex'] == "M"].index.tolist()
            for exc in excludefilters:
                if(exc==".."):
                    filterId=annowithloc[annowithloc['Y haplogroup'] == ".."].index.tolist()
                else:
                    filterId=annowithloc[annowithloc['Y haplogroup'].str.contains(exc, na=False)].index.tolist()
                malesId = [i for i in malesId if i not in filterId]
            annowithloc=annowithloc.filter(items = malesId, axis=0)
        else:
            annowithloc=annowithloc[annowithloc['Molecular Sex'] == "M"]
    else:
        annowithloc=CreateLocalityFile(annofilename,verbose=verbose,toCSV=toCSV)

    keepIDs=annowithloc.index.to_list()

    geno=geno[:,keepIDs]

    if(minSNPnbr>0):
        if(verbose):
            print("Filtering Individuals with too few SNPs")
        if(minSNPnbr>1):
            minSNPnbr_local=minSNPnbr
        else:
            minSNPnbr_local=math.floor(minSNPnbr*geno.shape[0])
        keep_indexes=np.count_nonzero(geno<3,axis=0)>minSNPnbr_local
        geno=geno[:,keep_indexes]
        annowithloc=annowithloc.loc[keep_indexes==True]

        if(verbose):
            print(str(annowithloc.shape[0])+" individuals were kept")

    if(to_numpy==False):
        if(verbose):
            print("Generating Pandas Dataframe with the proper SNP and individual indexes")
        df=pd.DataFrame(geno,columns=annowithloc['Genetic ID'],index=snp.filter(items=chr22dna,axis=0)[0])
        if(transpose):
            df=df.T
        if(toCSV):
            if(verbose):
                print("Exporting DNA CSV")
            df.to_csv(genofilename+"_dna.csv",sep=';')
    else:
        df=geno
        if(transpose):
            df=df.T
        if(toCSV):
            if(verbose):
                print("Exporting DNA CSV")
            np.savetxt(genofilename+"_dna.csv", geno.astype(int),fmt="%i", delimiter=";")

    if(verbose):
        print("DONE")

    return df,annowithloc

##### unpackYDNAfull(genofilename,snpfilename, indfilename="" , transpose=True,toCSV=False)
#
#    Extract the snp for chromosome Y from a given geno file as a pandas dataframe.
#    We suggest using unpackYDNAfromAnno() if you have the .anno file
#
#   parameters:
#       filename : path/filename for the .geno
#       snpfilename : path/filename for the .snp
#       indfilename : path/filename for the .ind
#       transpose : boolean parameter to transpose the final matrix or not
#       toCSV : boolean parameter, if set to true a CSV file with the resulting matrix is created in the same folder as the geno file
#
#   returns:
#       df : the created pandas dataframe
#
def unpackYDNAfull(genofilename,snpfilename, indfilename="" , transpose=True,toCSV=False):
    """
    从 .geno 文件中提取 Y 染色体 (chromosome 24) 的 SNP 信息。
        - 读取 .snp 文件识别 Y 染色体；
        - 解包相应基因型数据；
        - 可选读取 .ind 文件过滤男性；
        - 输出 pandas DataFrame 或导出 CSV。

    :param genofilename: .geno 文件路径
    :param snpfilename: .snp 文件路径
    :param indfilename: .ind 文件路径（可留空）
    :param transpose: 是否转置矩阵
    :param toCSV: 是否导出 CSV
    :return: Y 染色体基因型 DataFrame
    """
    geno_file, nind,nsnp,rlen=loadRawGenoFile(genofilename,True)
    geno=np.fromfile(genofilename, dtype='uint8')[rlen:]
    geno.shape=(nsnp,rlen)
    snp=pd.read_csv(snpfilename, sep=r"\s+", header=None)
    ydna=snp[snp[1] == 24].index.tolist()
    geno=unpackAndFilterSNPs(geno,ydna,nind)


    if(indfilename==""):
        df=pd.DataFrame(geno,index=snp.filter(items=ydna,axis=0)[0])
    else:
        ind=pd.read_csv(indfilename, sep=r"\s+", header=None)
        malesId=ind[ind[1] == "M"].index.tolist()
        geno=geno[:,malesId]
        df=pd.DataFrame(geno,columns=ind.filter(items=malesId,axis=0)[0],index=snp.filter(items=ydna,axis=0)[0])

    #df=df.replace(to_replace=[3], value=[""],regex=True) #missing values
    if(transpose):
        df=df.T

    if(toCSV):
        df.to_csv(genofilename+"_ydna.csv",sep=';')

    return df


##### unpackChromosome(genofilename,snpfilename, chrNbr, indfilename="" , transpose=True,toCSV=False)
#
#    Extract the snp for chromosome "chrNbr" from a given geno file as a pandas dataframe.
#    We suggest using the unpackChromosomefromAnno() if you have the .anno file
#
#   parameters:
#       filename : path/filename for the .geno
#       snpfilename : path/filename for the .snp
#       chrNbr : the numer of the chromosome you are interested in
#       indfilename : path/filename for the .ind
#       transpose : boolean parameter to transpose the final matrix or not
#       toCSV : boolean parameter, if set to true a CSV file with the resulting matrix is created in the same folder as the geno file
#
#   returns:
#       df : the created pandas dataframe
#
def unpackChromosome(genofilename,snpfilename, chrNbr, indfilename="" , transpose=True,toCSV=False):
    """
    从 .geno 文件中提取指定染色体 (chrNbr) 的 SNP 数据。
        - 通过 .snp 文件筛选目标染色体；
        - 读取并解码 SNP；
        - 可选读取 .ind 文件定义列名；
        - 支持输出 CSV 文件。

    :param genofilename: .geno 文件路径
    :param snpfilename: .snp 文件路径
    :param chrNbr: 染色体编号（1–24）
    :param indfilename: .ind 文件路径（可为空）
    :param transpose: 是否转置输出矩阵
    :param toCSV: 是否导出为 CSV 文件
    :return: 对应染色体的基因型 DataFrame
    """
    if(chrNbr!=24):
        geno_file, nind,nsnp,rlen=loadRawGenoFile(genofilename,True)
        geno=np.fromfile(genofilename, dtype='uint8')[rlen:]
        geno.shape=(nsnp,rlen)
        snp=pd.read_csv(snpfilename, sep=r"\s+", header=None)
        chrlist=snp[snp[1] == chrNbr].index.tolist()
        geno=unpackAndFilterSNPs(geno,chrlist,nind)

        if(indfilename==""):
            df=pd.DataFrame(geno,index=snp.filter(items=chrlist,axis=0)[0])
        else:
            ind=pd.read_csv(indfilename, sep=r"\s+", header=None)
            df=pd.DataFrame(geno,columns=ind[0],index=snp.filter(items=chrlist,axis=0)[0])


        #df=df.replace(to_replace=[3], value=[""],regex=True) #missing values
        if(transpose):
            df=df.T

        if(toCSV):
            df.to_csv(genofilename+"_dna.csv",sep=',')

    else:
        df=unpackYDNAfull(genofilename,snpfilename, indfilename, transpose,toCSV)

    return df


##### unpackChromosomefromAnno(genofilename,snpfilename, annofilename,chrNbr,transpose=True,toCSV=False)
#
#    Extract the snp for chromosome "chrNbr" from a given geno file as a pandas dataframe.
#    We suggest using the unpackYDNAfromAnno() for the Y chromosome
#
#   parameters:
#       filename : path/filename for the .geno
#       snpfilename : path/filename for the .snp
#       chrNbr : the numer of the chromosome you are interested in
#       annofilename : path/filename for the .anno
#       transpose : boolean parameter to transpose the final matrix or not
#       toCSV : boolean parameter, if set to true a CSV file with the resulting matrix is created in the same folder as the geno file
#
#   returns:
#       df : the created pandas dataframe
#      
def unpackChromosomefromAnno(genofilename,snpfilename, annofilename,chrNbr,transpose=True,toCSV=False):
    """
    通过 .anno 文件提取指定染色体的 SNP 数据。
        - 根据 .snp 文件定位目标染色体；
        - 读取并解包 .geno；
        - 使用 .anno 中的样本信息作为列索引；
        - 可导出 CSV 文件。

    :param genofilename: .geno 文件路径
    :param snpfilename: .snp 文件路径
    :param annofilename: .anno 文件路径
    :param chrNbr: 染色体编号
    :param transpose: 是否转置结果
    :param toCSV: 是否导出为 CSV
    :return: 基因型 DataFrame
    """
    geno_file, nind,nsnp,rlen=loadRawGenoFile(genofilename,True)
    geno=np.fromfile(genofilename, dtype='uint8')[rlen:]
    geno.shape=(nsnp,rlen)
    snp=pd.read_csv(snpfilename, sep=r"\s+", header=None)
    chrlist=snp[snp[1] == chrNbr].index.tolist()
    geno=unpackAndFilterSNPs(geno,chrlist,nind)
    #if(chrNbr==90):
        #geno[geno==2]=1                                   #you can have only one version of mtDNA


    
    anno=pd.read_csv(annofilename, sep="	",low_memory=False)       
    df=pd.DataFrame(geno,columns=anno['Genetic ID'],index=snp.filter(items=chrlist,axis=0)[0])
    #df=df.replace(to_replace=[3], value=[""],regex=True) #missing values     
        
    if(transpose):
        df=df.T        
        
    if(toCSV):
        df.to_csv(genofilename+"_dna.csv",sep=';')     


##### FilterYhaplIndexes(pdAnno,includefilters=None,excludefilters=["na"," ",".."])
#
#    A subfunction used when browsing through yDNA : it removes non male individuals and allow filters on haplogroups
#    By default, the filter remove individual whose haplogroup is set to "..", or contain spaces, or the string "na"
#
#   parameters:
#       pdAnno : a panda dataframe of the .anno file
#       includefilters : an array of haplogroups (or lack of) to select. Can be set to None
#       excludefilters : an array of haplogroups (or lack of) to filter. Can be set to None
#
#   returns:
#       malesId : the index list of individuals from the .anno file to be kept
#   
def FilterYhaplIndexes(pdAnno,includefilters=None,excludefilters=["na"," ",".."]):
    """
    过滤 Y 染色体样本索引。
        - 仅保留男性个体；
        - 可定义包含或排除的单倍群关键字；
        - 默认排除未知或无效值。

    :param pdAnno: 从 .anno 文件读取的 DataFrame
    :param includefilters: 包含的单倍群列表（可为 None）
    :param excludefilters: 排除的单倍群关键字（默认 ["na"," ",".."]）
    :return: 保留的男性样本索引列表
    """
    if(includefilters!=None):
        malesId=[]
        for inc in includefilters:
            filterId=pdAnno[pdAnno['Y haplogroup (manual curation in ISOGG format)'].str.contains(inc, na=False)].index.tolist()
            newIds= [i for i in filterId if i not in malesId]
            malesId.extend(newIds)
    else:
        malesId=pdAnno[pdAnno['Molecular Sex'] == "M"].index.tolist()
        
    if(excludefilters!=None and len(malesId)>0):
        for exc in excludefilters:
            if(exc==".."):
                filterId=pdAnno[pdAnno['Y haplogroup (manual curation in ISOGG format)'] == ".."].index.tolist()
            else:
                filterId=pdAnno[pdAnno['Y haplogroup (manual curation in ISOGG format)'].str.contains(exc, na=False)].index.tolist()
            malesId = [i for i in malesId if i not in filterId]
   
    return malesId


##### ExtractYHaplogroups(annofile,separator="	",includefilters=None,excludefilters=None)
#
#    A function used to extract a list of y Haplogroups from a .anno file
#
#   parameters:
#       annofilename : path/filename for the .anno
#       separator : separator used in the .anno file
#       includefilters : an array of haplogroups (or lack of) to select. Can be set to None
#       excludefilters : an array of haplogroups (or lack of) to filter. Can be set to None
#
#   returns:
#       ygroups : the list of haplogroups matching the kept indexes
#       malesId : the index list of individuals from the .anno file to be kept
#  
def ExtractYHaplogroups(annofile,separator="	",includefilters=None,excludefilters=None):
    """
    从 .anno 文件中提取 Y 染色体单倍群信息。
        - 通过 FilterYhaplIndexes() 获取男性样本索引；
        - 提取对应单倍群字段；
        - 返回单倍群列表与索引。

    :param annofile: .anno 文件路径
    :param separator: 文件分隔符（默认制表符）
    :param includefilters: 包含过滤器
    :param excludefilters: 排除过滤器
    :return: (ygroups, malesId)
             - ygroups: 单倍群序列
             - malesId: 样本索引列表
    """
    anno=pd.read_csv(annofile, sep=separator,low_memory=False)
    malesId=FilterYhaplIndexes(anno,includefilters,excludefilters)
    ygroups=anno.filter(items=malesId,axis=0)['Y haplogroup (manual curation in ISOGG format)']
    return ygroups,malesId


##### unpackYDNAfromAnno(genofilename,snpfilename,annofilename,excludefilters=None,transpose=True,toCSV=False)
#
#    Extract the snp for chromosome Y from a given geno file as a pandas dataframe.
#    We suggest using the unpackYDNAfromAnno() for the Y chromosome
#
#   parameters:
#       filename : path/filename for the .geno
#       snpfilename : path/filename for the .snp
#       annofilename : path/filename for the .anno
#       includefilters : an array of haplogroups (or lack of) to select. Can be set to None
#       excludefilters : an array of haplogroups (or lack of) to filter. Can be set to None
#       transpose : boolean parameter to transpose the final matrix or not
#       toCSV : boolean parameter, if set to true a CSV file with the resulting matrix is created in the same folder as the geno file
#
#   returns:
#       df : the created pandas dataframe
#  
def unpackYDNAfromAnno(genofilename,snpfilename, annofilename,includefilters=None,excludefilters=None,transpose=True,toCSV=False):
    """
    基于 .anno 文件提取 Y 染色体的 SNP 基因型数据。
        - 读取 .geno、.snp 与 .anno；
        - 筛选 Y 染色体（chromosome 24）；
        - 过滤个体（男性与指定单倍群）；
        - 生成 pandas DataFrame，可导出 CSV。

    :param genofilename: .geno 文件路径
    :param snpfilename: .snp 文件路径
    :param annofilename: .anno 文件路径
    :param includefilters: 包含的单倍群过滤器
    :param excludefilters: 排除的单倍群过滤器
    :param transpose: 是否转置结果矩阵
    :param toCSV: 是否导出 CSV
    :return: Y 染色体基因型 DataFrame
    """
    geno_file, nind,nsnp,rlen=loadRawGenoFile(genofilename,True)
    geno=np.fromfile(genofilename, dtype='uint8')[rlen:]
    geno.shape=(nsnp,rlen)
    
    snp=pd.read_csv(snpfilename, sep=r"\s+", header=None)
    ydna=snp[snp[1] == 24].index.tolist()
    geno=unpackAndFilterSNPs(geno,ydna,nind)
    #geno[geno==2]=1                                    #you can have only one Y chromosome
        
    anno=pd.read_csv(annofilename, sep="	",low_memory=False)
    malesId=FilterYhaplIndexes(anno,includefilters,excludefilters)
    geno=geno[:,malesId]             
    
    df=pd.DataFrame(geno,columns=anno.filter(items=malesId,axis=0)['Genetic ID'],index=snp.filter(items=ydna,axis=0)[0])
    #df=df.replace(to_replace=[3], value=[""],regex=True) #missing values    
        
    if(transpose):
        df=df.T        
        
    if(toCSV):
        df.to_csv(genofilename+"_ydna.csv",sep=';')  
        
    return df