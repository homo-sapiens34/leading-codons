import scipy.stats as st 
from Bio.Seq import Seq
from Bio import SeqIO
from Bio.Alphabet import IUPAC
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import random
from scipy.stats.mstats import mquantiles
import pickle
from scipy.stats import chi2_contingency
import numpy as np
import pandas as pd
import math
from matplotlib.patches import Rectangle

SPEARMAN_R_THRESHOLD = 0.8
FLAG_USE_P_V_THRESHOLD_FOR_REAL = False
FLAG_USE_P_V_THRESHOLD_FOR_SIMULATED = False
HEATMAP_P_V_THRESHOLD = 0.01
COEFF = 10**3 

def get_file_name(name):
    return "../CodonRamp_Final_" + name + ".tab" 

def get_class_5_merge_6(p):
    for i in range(1,7):
        if p>(i-1)*2 and p<=i*2:
            if i == 6:
                return 5
            return i
        
def import_data(fname):
    class2ss = {}
    all_ss2p = {}
    class2p = {}
    all_p = []
    ss2number = {}
    num2class = {}

    with open(fname) as f:
        f.readline()
        for i,line in enumerate(f):
            s = line.strip().split("\t")
            ss = s[0]
            p = float(s[1])
            all_ss2p[ss] = p
            cl = get_class_5_merge_6(p)
            if not cl in class2p:
                class2p[cl] = []
            class2p[cl].append(p)
            all_p.append(p)
            if not cl in class2ss:
                class2ss[cl] = []
            class2ss[cl].append(ss)
            ss2number[ss] = i 
            num2class[i] = cl
        
    return class2ss, all_ss2p, class2p, all_p, ss2number, num2class

def get_all_codons(class2ss):
    codons = {}
    for c in sorted(class2ss):
        ss_ar = class2ss[c]
        for ss in ss_ar:
            for i in range(10):
                co = ss[i*3:(i*3+3)]
                if not co in codons:
                    codons[co] = 1
    cods = []
    for co in codons:
        cods.append(co)
    return cods

def calc_tai(ss):
    ws = 1
    for i in range(0, len(ss), 3):
        cod = ss[i:i+3]
        w = codon2w[cod]
        ws *= w
    return ws**0.1

def get_real_codon_fr(class2ss):
    class2real_codons = {}
    all_codons = get_all_codons(class2ss)
    for c in sorted(class2ss):
        class2real_codons[c] = {}
        for i in range(10):
            class2real_codons[c][i] = {}
            sum_q = 0.0
            p0 = i*3
            for ss in class2ss[c]:
                codon = ss[p0:p0+3]
                if not codon in class2real_codons[c][i]:
                    class2real_codons[c][i][codon] = 0
                class2real_codons[c][i][codon] += 1
                sum_q +=1

            for codon in class2real_codons[c][i]:
                class2real_codons[c][i][codon] /= sum_q
                
            for codon in all_codons:
                if not codon in class2real_codons[c][i]:
                    class2real_codons[c][i][codon] = 0.0
    return class2real_codons


def is_stop(codon):
    if codon == "TAA" or codon == "TAG" or codon == "TGA":
        return True
    return False

def count_corr_coef(cls, class_v):
    pearson = st.pearsonr(cls, class_v)
    spearman = st.spearmanr(cls, class_v)
    return {"pearson":pearson[0], "spearman":spearman[0]}

def count_effect(cls, class_v, codon, pos):
    x = np.array(cls)
    y = np.array(class_v)
    return st.linregress(cls,class_v)[0]

def clean_stops(class2ss):
    class2ss_new = {}
    for cl in class2ss:
        class2ss_new[cl] = []
        for ss in class2ss[cl]:
            is_bad = False
            for i in range(10):
                p0 = i*3
                codon = ss[p0:p0+3]
                if is_stop(codon):
                    is_bad = True
            if not is_bad:
                class2ss_new[cl].append(ss)
    return class2ss_new

def get_aa_for_codon(codon):
    coding_dna = Seq(codon, IUPAC.unambiguous_dna)
    tr = coding_dna.translate()
    return str(tr)

def get_codons_for_aa(aa, aa_dict):
    return aa_dict[aa]

def get_ef_array(full_name2r, name, pos, codon):
    ef = []
    for v in full_name2r[name][pos][codon]:
        if abs(v[0]["spearman"]) > SPEARMAN_R_THRESHOLD: 
            ef.append(v[1])
        else:
            if not FLAG_USE_P_V_THRESHOLD_FOR_REAL:
                ef.append(v[1])
    return ef
    
def find_nearest(r, X, F):
    i = 0
    for x in X:
        if x <= r:
            i += 1
        else:
            break 
    if i == len(X):
        return 1
    if i == 0:
        return 0
    return F[i-1]   

def get_effect_p_value(e, full_name2r, name, pos, codon):
    Z = get_ef_array(full_name2r, name, pos, codon)
    H,X1 = np.histogram(Z, bins = len(Z), normed = True)
    dx = X1[1] - X1[0]
    F1 = np.cumsum(H)*dx
    p = find_nearest(e, X1[1:], F1) 
    return p

def get_effect_z_score(e, full_name2r, name, pos, codon):
    perm = get_ef_array(full_name2r, name, pos, codon)    
    m = np.mean(perm)
    std = np.std(perm)
    z = (e-m)/std  
    return z

def get_mon_p_value(e, s):
    Z = s
    H,X1 = np.histogram(Z, bins = len(Z), normed = True)
    dx = X1[1] - X1[0]
    F1 = np.cumsum(H)*dx
    p = find_nearest(e, X1[1:], F1) 
    return p

def reorganize_array2(pos2codon2ef):
    codon2ef = {}
    mask_f = {}
    for pos in range(10):
        for codon in sorted(pos2codon2ef[pos]):
            aa = get_aa_for_codon(codon)
            label = aa + " " + codon.replace("T", "U")
            if not label in codon2ef: 
                codon2ef[label] = []
                mask_f[label] = []
            pv = pos2codon2ef[pos][codon][1]
            ef = pos2codon2ef[pos][codon][0]
            if (ef < 0 and pv < HEATMAP_P_V_THRESHOLD) or (ef > 0 and pv > 1 - HEATMAP_P_V_THRESHOLD):
                codon2ef[label].append(ef*COEFF)
                mask_f[label].append(False)
            else:
                codon2ef[label].append(ef*COEFF)
                mask_f[label].append(True)
    return codon2ef, mask_f

def add_rectangles(codon2ef, mask_f, ax, i, pos2codon2ef):
    sorted_labels = sorted(codon2ef.keys())
    for j, label in enumerate(sorted_labels):
        codon = label.split()[1].replace("U", "T")
        for pos in range(len(codon2ef[sorted_labels[0]])):
            if not mask_f[label][pos]:
                ax[i].add_patch(Rectangle((pos, j), 1, 1, fill=False, edgecolor='black', lw=1.5))
                if abs(pos2codon2ef[pos][codon][2]["spearman"]) < 0.6: ##это про что-то другое
                    ax[i].add_patch(Rectangle((pos, j), 1, 1, fill=False, edgecolor='blue', lw=1))

def reorganize_array_z(pos2codon2ef, pos2codon2z):
    codon2z = {}
    mask_f = {}
    for pos in range(10):
        for codon in sorted(pos2codon2ef[pos]):
            aa = get_aa_for_codon(codon)
            label = aa + " " + codon.replace("T", "U")
            if not label in codon2z: 
                codon2z[label] = []
                mask_f[label] = []
            pv = pos2codon2ef[pos][codon][1]
            ef = pos2codon2ef[pos][codon][0]
            z = pos2codon2z[pos][codon][0]
            if (ef < 0 and pv < HEATMAP_P_V_THRESHOLD) or (ef > 0 and pv > 1 - HEATMAP_P_V_THRESHOLD):
                codon2z[label].append(z)
                mask_f[label].append(False)
            else:
                if ef == 0:
                    codon2z[label].append(0.0) 
                else:
                    codon2z[label].append(z)
                mask_f[label].append(True)
    return codon2z, mask_f

def get_best(pos2codon2z):
    poses = list(range(10))
    best_codons = []
    worst_codons = []
    for pos in poses:
        codons = list(pos2codon2z[pos])
        codons = sorted(codons, key=lambda x: pos2codon2z[pos][codon])
        best_codons.append(codons[-1])
        worst_codons.append(codons[0])
    return [best_codons, worst_codons]

def sort_codons(name2pos2codon2z_score):
    name2pos2sorted_codons = {}
    for name in name2pos2codon2z_score:
        name2pos2sorted_codons[name] = {}
        for pos in sorted(name2pos2codon2z_score[name]):
            #print(pos)
            codons = list(name2pos2codon2z_score[name][pos].keys())
            codons = sorted(codons, key=lambda x: name2pos2codon2z_score[name][pos][x][0]) #ascending -- bad codons first
            info = []
            for c in codons:
                info.append([c, name2pos2codon2z_score[name][pos][c][0]])
            name2pos2sorted_codons[name][pos] = info
    return name2pos2sorted_codons

