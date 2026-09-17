import csv, collections, random, sys, time
import numpy as np
from numba import njit
from scipy.optimize import linear_sum_assignment
import exp, phon

@njit(cache=True)
def estep2(cip_flat,cip_off,ref_flat,ref_off,pair_lang,lang_w,sub_score,gap,T,S,nlang,relpow):
    counts=np.zeros((T,S))
    lang_cov=np.zeros(nlang); lang_cnt=np.zeros(nlang)
    npair=cip_off.shape[0]-1
    MAXN=40; MAXM=60
    D=np.empty((MAXN+1,MAXM+1)); B=np.empty((MAXN+1,MAXM+1),dtype=np.int8)
    mt=np.empty(MAXN+MAXM,dtype=np.int64); ms=np.empty(MAXN+MAXM,dtype=np.int64)
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
                up=D[i-1,j]+gap; left=D[i,j-1]+gap
                if diag>=up and diag>=left: D[i,j]=diag; B[i,j]=0
                elif up>=left: D[i,j]=up; B[i,j]=1
                else: D[i,j]=left; B[i,j]=2
        i=n; j=m; matched=0
        while i>0 or j>0:
            if i>0 and j>0 and B[i,j]==0:
                mt[matched]=cip_flat[cs+i-1]; ms[matched]=ref_flat[rs+j-1]; matched+=1; i-=1; j-=1
            elif i>0 and (j==0 or B[i,j]==1): i-=1
            else: j-=1
        mx=n if n>m else m
        rel=matched/mx
        pw=w*(rel**relpow)
        for a in range(matched):
            counts[mt[a],ms[a]]+=pw
        lang_cov[lg]+=rel; lang_cnt[lg]+=1.0
    return counts,lang_cov,lang_cnt

def decipher2(test_rows, support_by_concept, seg_pool, n_iter=10, wpow=3.0, gap=-3.0,
              relpow=0.0, smooth=False, tau=0.35, self_boost=2.0, ksmooth_final=True,
              return_counts=False):
    tok_vocab=sorted({t for _,toks in test_rows for t in toks})
    tok_id={t:i for i,t in enumerate(tok_vocab)}
    seg_vocab=sorted(seg_pool); seg_id={s:i for i,s in enumerate(seg_vocab)}
    T=len(tok_vocab); S=len(seg_vocab)
    cip_flat,cip_off,ref_flat,ref_off,pair_lang,langs,lang_idx=exp.build_pairs(test_rows,support_by_concept,seg_id,tok_id)
    nlang=len(langs); lang_w=np.ones(nlang)
    K=phon.build_kernel(seg_vocab,tau=tau,self_boost=self_boost) if smooth else None
    C=np.ones((T,S))*0.01
    for p in range(cip_off.shape[0]-1):
        ci=cip_flat[cip_off[p]:cip_off[p+1]]; rf=ref_flat[ref_off[p]:ref_off[p+1]]
        w=1.0/(len(ci)*len(rf))
        for t in ci: C[t,rf]+=w
    sub=np.log(C/C.sum(1,keepdims=True)+1e-9)
    counts=None
    for it in range(n_iter):
        counts,lcov,lcnt=estep2(cip_flat,cip_off,ref_flat,ref_off,pair_lang,lang_w,sub,gap,T,S,nlang,relpow)
        cs=counts+1e-6
        if smooth: cs=cs@K
        sub=np.log(cs/cs.sum(1,keepdims=True))
        cov=np.where(lcnt>0,lcov/np.maximum(lcnt,1),0.0)
        mx=cov.max() if cov.size else 1.0
        if mx>0: lang_w=(cov/mx)**wpow
    scoreTS=counts+1e-9
    if smooth and ksmooth_final: scoreTS=scoreTS@K
    r,c=linear_sum_assignment(-scoreTS)
    tokmap={tok_vocab[ri]:seg_vocab[ci] for ri,ci in zip(r,c)}
    if return_counts:
        return tokmap, tok_vocab, tok_id, seg_vocab, counts
    return tokmap

def make_support(exclude, restrict='uralic'):
    if restrict=='uralic': pool=[l for l in exp.URAL if l!=exclude]
    else: pool=[l for l in exp.LANG_FAM if l!=exclude]
    support=collections.defaultdict(list); seg_pool=set()
    for lg in pool:
        for con,segs in exp.by_lang[lg]:
            support[con].append((lg,segs)); seg_pool.update(segs)
    return support, seg_pool

def run_pseudo2(target_lang, **kw):
    words=exp.by_lang[target_lang]
    segset=sorted({s for _,segs in words for s in segs})
    rng=random.Random(12345); perm=segset[:]; rng.shuffle(perm)
    seg2tok={s:"x%d"%i for i,s in enumerate(perm)}
    test_rows=[]; true_words=[]
    for con,segs in words:
        test_rows.append((con,[seg2tok[s] for s in segs])); true_words.append(segs)
    support,seg_pool=make_support(target_lang)
    tokmap=decipher2(test_rows,support,seg_pool,**kw)
    pred=[[tokmap.get(t,'?') for t in toks] for _,toks in test_rows]
    sc=exp.score_pred(pred,true_words)
    return sc

if __name__=='__main__':
    targets=['fin','ekk','krl','olo','vep','myv','udm','mhr','sme','hun','yrk']
    configs={
      'A_base':dict(relpow=0.0,smooth=False),
      'B_smooth':dict(relpow=0.0,smooth=True),
      'C_rel':dict(relpow=1.0,smooth=False),
      'D_both':dict(relpow=1.0,smooth=True),
      'D2_rel2':dict(relpow=2.0,smooth=True),
    }
    which=sys.argv[1].split(',') if len(sys.argv)>1 else list(configs)
    res={}
    for name in which:
        cfg=configs[name]; scs=[]
        t0=time.time()
        for tg in targets:
            scs.append(run_pseudo2(tg,n_iter=10,wpow=3.0,gap=-3.0,**cfg))
        res[name]=scs
        fin=np.mean(scs[:5]); allm=np.mean(scs)
        print("%-9s finnicMEAN=%.4f allMEAN=%.4f  per:%s [%.0fs]"%(name,fin,allm,
              ' '.join('%s=%.3f'%(t,s) for t,s in zip(targets,scs)),time.time()-t0))
