import csv, collections, random, sys, math, time
import numpy as np
from numba import njit
from scipy.optimize import linear_sum_assignment

def load_train(path='train.csv'):
    rows=[]
    with open(path,encoding='utf-8') as f:
        for row in csv.DictReader(f):
            rows.append((row['language'],row['family'],row['subfamily'],row['concept'],row['ipa'].split()))
    return rows

TRAIN=load_train()
LANG_FAM={}; LANG_SUB={}
by_lang=collections.defaultdict(list)
for lg,fam,sub,con,seg in TRAIN:
    LANG_FAM[lg]=fam; LANG_SUB[lg]=sub
    by_lang[lg].append((con,seg))
URAL=[l for l in LANG_FAM if LANG_FAM[l]=='Uralic']

def lev(a,b):
    n,m=len(a),len(b)
    if n==0: return m
    if m==0: return n
    prev=list(range(m+1))
    for i in range(1,n+1):
        cur=[i]+[0]*m; ai=a[i-1]
        for j in range(1,m+1):
            cur[j]=min(prev[j]+1,cur[j-1]+1,prev[j-1]+(0 if ai==b[j-1] else 1))
        prev=cur
    return prev[m]

def score_pred(pred_words, true_words):
    tot=0.0
    for p,t in zip(pred_words,true_words):
        if len(p)==0 and len(t)==0: tot+=1.0; continue
        d=lev(p,t); m=max(len(p),len(t))
        tot+= (1.0-d/m) if m>0 else 1.0
    return tot/len(true_words)

@njit(cache=True)
def estep(cip_flat,cip_off,ref_flat,ref_off,pair_lang,lang_w,sub_score,gap,T,S,nlang):
    counts=np.zeros((T,S))
    lang_cov=np.zeros(nlang)
    lang_cnt=np.zeros(nlang)
    npair=cip_off.shape[0]-1
    MAXN=40; MAXM=60
    D=np.empty((MAXN+1,MAXM+1))
    B=np.empty((MAXN+1,MAXM+1),dtype=np.int8)
    for p in range(npair):
        lg=pair_lang[p]; w=lang_w[lg]
        cs=cip_off[p]; ce=cip_off[p+1]; n=ce-cs
        rs=ref_off[p]; re=ref_off[p+1]; m=re-rs
        if n>MAXN or m>MAXM or n==0 or m==0: continue
        D[0,0]=0.0
        for i in range(1,n+1): D[i,0]=D[i-1,0]+gap
        for j in range(1,m+1): D[0,j]=D[0,j-1]+gap
        for i in range(1,n+1):
            ti=cip_flat[cs+i-1]
            for j in range(1,m+1):
                sj=ref_flat[rs+j-1]
                diag=D[i-1,j-1]+sub_score[ti,sj]
                up=D[i-1,j]+gap
                left=D[i,j-1]+gap
                if diag>=up and diag>=left:
                    D[i,j]=diag; B[i,j]=0
                elif up>=left:
                    D[i,j]=up; B[i,j]=1
                else:
                    D[i,j]=left; B[i,j]=2
        # traceback
        i=n; j=m; matched=0
        while i>0 or j>0:
            if i>0 and j>0 and B[i,j]==0:
                ti=cip_flat[cs+i-1]; sj=ref_flat[rs+j-1]
                counts[ti,sj]+=w; matched+=1; i-=1; j-=1
            elif i>0 and (j==0 or B[i,j]==1):
                i-=1
            else:
                j-=1
        mx=n if n>m else m
        lang_cov[lg]+=matched/mx
        lang_cnt[lg]+=1.0
    return counts,lang_cov,lang_cnt

def build_pairs(test_rows, support_by_concept, seg_id, tok_id):
    langs_set=set()
    plist=[]
    for con,toks in test_rows:
        ci=[tok_id[t] for t in toks]
        for lg,segs in support_by_concept.get(con,[]):
            rf=[seg_id[s] for s in segs if s in seg_id]
            if rf:
                plist.append((ci,rf,lg)); langs_set.add(lg)
    langs=sorted(langs_set); lang_idx={l:i for i,l in enumerate(langs)}
    cip_flat=[]; cip_off=[0]; ref_flat=[]; ref_off=[0]; pair_lang=[]
    for ci,rf,lg in plist:
        cip_flat.extend(ci); cip_off.append(len(cip_flat))
        ref_flat.extend(rf); ref_off.append(len(ref_flat))
        pair_lang.append(lang_idx[lg])
    return (np.array(cip_flat,np.int32),np.array(cip_off,np.int64),
            np.array(ref_flat,np.int32),np.array(ref_off,np.int64),
            np.array(pair_lang,np.int64),langs,lang_idx)

def decipher(test_rows, support_by_concept, all_segs_pool, n_iter=12, wpow=3.0, gap=-3.0, verbose=False):
    tok_vocab=sorted({t for _,toks in test_rows for t in toks})
    tok_id={t:i for i,t in enumerate(tok_vocab)}
    seg_vocab=sorted(all_segs_pool)
    seg_id={s:i for i,s in enumerate(seg_vocab)}
    T=len(tok_vocab); S=len(seg_vocab)
    cip_flat,cip_off,ref_flat,ref_off,pair_lang,langs,lang_idx=build_pairs(test_rows,support_by_concept,seg_id,tok_id)
    nlang=len(langs)
    lang_w=np.ones(nlang)
    # init co-occurrence
    C=np.ones((T,S))*0.01
    npair=cip_off.shape[0]-1
    for p in range(npair):
        ci=cip_flat[cip_off[p]:cip_off[p+1]]; rf=ref_flat[ref_off[p]:ref_off[p+1]]
        w=1.0/(len(ci)*len(rf))
        for t in ci:
            C[t,rf]+=w
    sub=np.log(C/C.sum(1,keepdims=True)+1e-9)
    counts=None
    for it in range(n_iter):
        counts,lcov,lcnt=estep(cip_flat,cip_off,ref_flat,ref_off,pair_lang,lang_w,sub,gap,T,S,nlang)
        sub=np.log((counts+1e-6)/(counts+1e-6).sum(1,keepdims=True))
        cov=np.where(lcnt>0,lcov/np.maximum(lcnt,1),0.0)
        mx=cov.max() if cov.size else 1.0
        if mx>0: lang_w=(cov/mx)**wpow
        if verbose:
            order=np.argsort(-cov)[:6]
            print("  iter%d topcov:"%it,[(langs[i],round(cov[i],3)) for i in order])
    scoreTS=counts
    r,c=linear_sum_assignment(-scoreTS)
    tokmap={}
    for ri,ci in zip(r,c):
        tokmap[tok_vocab[ri]]=seg_vocab[ci]
    return tokmap

def run_pseudo(target_lang, n_iter=10, restrict_uralic=True, wpow=3.0, gap=-3.0, verbose=False):
    words=by_lang[target_lang]
    segset=sorted({s for _,segs in words for s in segs})
    rng=random.Random(12345)
    perm=segset[:]; rng.shuffle(perm)
    seg2tok={s:"x%d"%i for i,s in enumerate(perm)}
    test_rows=[]; true_words=[]
    for con,segs in words:
        test_rows.append((con,[seg2tok[s] for s in segs])); true_words.append(segs)
    pool_langs=[l for l in (URAL if restrict_uralic else list(LANG_FAM)) if l!=target_lang]
    support=collections.defaultdict(list); seg_pool=set()
    for lg in pool_langs:
        for con,segs in by_lang[lg]:
            support[con].append((lg,segs)); seg_pool.update(segs)
    tokmap=decipher(test_rows,support,seg_pool,n_iter=n_iter,wpow=wpow,gap=gap,verbose=verbose)
    pred_words=[[tokmap.get(t,'?') for t in toks] for con,toks in test_rows]
    sc=score_pred(pred_words,true_words)
    correct=0; total=0
    for (con,toks),tw in zip(test_rows,true_words):
        for t,tr in zip(toks,tw):
            total+=1
            if tokmap.get(t)==tr: correct+=1
    return sc, correct/total

if __name__=='__main__':
    targets=sys.argv[1].split(',') if len(sys.argv)>1 else ['fin','krl','hun','yrk','sme','udm','mdf','mhr']
    t0=time.time()
    scs=[]
    for tg in targets:
        sc,acc=run_pseudo(tg,n_iter=10,verbose=(tg==targets[0]))
        scs.append(sc)
        print("TARGET=%s sub=%-12s score=%.4f segacc=%.4f  [%.1fs]"%(tg,LANG_SUB[tg],sc,acc,time.time()-t0))
    print("MEAN score=%.4f"%(sum(scs)/len(scs)))
