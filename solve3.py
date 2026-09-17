import csv, collections, random, sys, time
import numpy as np
from numba import njit
from scipy.optimize import linear_sum_assignment
import exp, solve2, phon

@njit(cache=True)
def estep_anchored(cip_flat,cip_off,ref_flat,ref_off,pair_lang,lang_w,sub_score,gap,T,S,nlang,
                   tokpred,tok_isvow,skelpow):
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
        # consonant-skeleton agreement
        ncons=0; cmatch=0
        for a in range(matched):
            t=mt[a]
            if tok_isvow[t]==0:
                ncons+=1
                if ms[a]==tokpred[t]: cmatch+=1
        # also count total consonant tokens in cipher word for denom
        ncons_word=0
        for i2 in range(n):
            if tok_isvow[cip_flat[cs+i2]]==0: ncons_word+=1
        denom=ncons_word if ncons_word>0 else 1
        skel=cmatch/denom
        pw=w*(skel**skelpow)
        for a in range(matched):
            counts[mt[a],ms[a]]+=pw
        mx=n if n>m else m
        lang_cov[lg]+=matched/mx; lang_cnt[lg]+=1.0
    return counts,lang_cov,lang_cnt

def category_assign(score, tok_isvow, seg_isvow, tok_vocab, seg_vocab):
    T,S=score.shape
    tokmap={}
    for cat in [0,1]:  # 0=cons,1=vow
        tks=[i for i in range(T) if tok_isvow[i]==cat]
        sgs=[j for j in range(S) if seg_isvow[j]==cat]
        if not tks: continue
        if not sgs:  # fallback: allow any
            sgs=list(range(S))
        sub=score[np.ix_(tks,sgs)]
        r,c=linear_sum_assignment(-sub)
        for ri,ci in zip(r,c):
            tokmap[tok_vocab[tks[ri]]]=seg_vocab[sgs[ci]]
        # leftover tokens (more tokens than segs in category): argmax
        assigned=set(int(x) for x in c)
        for ri in range(len(tks)):
            if ri not in assigned and tok_vocab[tks[ri]] not in tokmap:
                j=int(np.argmax(score[tks[ri]]))
                tokmap[tok_vocab[tks[ri]]]=seg_vocab[j]
    return tokmap

def decipher3(test_rows, support_by_concept, seg_pool, n_iter=12, wpow=12.0, gap=-3.0,
              relpow=0.5, anchor_iters=6, skelpow=3.0, catconstrain=True):
    tok_vocab=sorted({t for _,toks in test_rows for t in toks})
    tok_id={t:i for i,t in enumerate(tok_vocab)}
    seg_vocab=sorted(seg_pool); seg_id={s:i for i,s in enumerate(seg_vocab)}
    T=len(tok_vocab); S=len(seg_vocab)
    cf,co,rf,ro,pl,langs,li=exp.build_pairs(test_rows,support_by_concept,seg_id,tok_id)
    nlang=len(langs); lang_w=np.ones(nlang)
    seg_isvow=np.array([1 if phon.feat_vec(s)[4]>0.5 else 0 for s in seg_vocab],dtype=np.int64)
    C=np.ones((T,S))*0.01
    for p in range(co.shape[0]-1):
        ci=cf[co[p]:co[p+1]]; rr=rf[ro[p]:ro[p+1]]; w=1.0/(len(ci)*len(rr))
        for t in ci: C[t,rr]+=w
    sub=np.log(C/C.sum(1,keepdims=True)+1e-9)
    counts=None
    # phase 1: standard EM
    for it in range(n_iter):
        counts,lcov,lcnt=solve2.estep2(cf,co,rf,ro,pl,lang_w,sub,gap,T,S,nlang,relpow)
        cs=counts+1e-6; sub=np.log(cs/cs.sum(1,keepdims=True))
        cov=np.where(lcnt>0,lcov/np.maximum(lcnt,1),0.0); mx=cov.max()
        if mx>0: lang_w=(cov/mx)**wpow
    # token vowel/consonant category from vmass
    vmass=(counts*seg_isvow[None,:]).sum(1)/(counts.sum(1)+1e-9)
    tok_isvow=(vmass>0.5).astype(np.int64)
    # phase 2: consonant-anchored refinement
    for it in range(anchor_iters):
        # current predicted seg per token (within-category argmax to keep skeleton clean)
        tokpred=np.zeros(T,dtype=np.int64)
        for t in range(T):
            row=counts[t].copy()
            mask=(seg_isvow==tok_isvow[t])
            if mask.any():
                row=np.where(mask,row,-1.0)
            tokpred[t]=int(np.argmax(row))
        counts,lcov,lcnt=estep_anchored(cf,co,rf,ro,pl,lang_w,sub,gap,T,S,nlang,
                                        tokpred,tok_isvow,skelpow)
        cs=counts+1e-6; sub=np.log(cs/cs.sum(1,keepdims=True))
        cov=np.where(lcnt>0,lcov/np.maximum(lcnt,1),0.0); mx=cov.max()
        if mx>0: lang_w=(cov/mx)**wpow
    score=counts+1e-9
    if catconstrain:
        tokmap=category_assign(score,tok_isvow,seg_isvow,tok_vocab,seg_vocab)
    else:
        r,c=linear_sum_assignment(-score)
        tokmap={tok_vocab[ri]:seg_vocab[ci] for ri,ci in zip(r,c)}
    return tokmap

def run_pseudo3(target, **kw):
    words=exp.by_lang[target]
    segset=sorted({s for _,segs in words for s in segs})
    rng=random.Random(12345); perm=segset[:]; rng.shuffle(perm)
    s2t={s:"x%d"%i for i,s in enumerate(perm)}
    tr=[(con,[s2t[s] for s in segs]) for con,segs in words]
    true=[segs for con,segs in words]
    support,seg_pool=solve2.make_support(target)
    tm=decipher3(tr,support,seg_pool,**kw)
    pred=[[tm.get(t,'?') for t in toks] for _,toks in tr]
    return exp.score_pred(pred,true)

if __name__=='__main__':
    TARG=['fin','ekk','krl','olo','vep','sme','smj','sjd','sms','myv','udm']
    import itertools
    t0=time.time()
    def show(name,**kw):
        scs=[run_pseudo3(tg,**kw) for tg in TARG]
        print("%-16s finnic=%.4f hard=%.4f all=%.4f | %s"%(name,
            np.mean(scs[:5]),np.mean(scs[5:9]),np.mean(scs),
            ' '.join('%s=%.3f'%(t,s) for t,s in zip(TARG,scs))))
        return scs
    show("v1_baseline",n_iter=12,anchor_iters=0,skelpow=3.0,catconstrain=False)
    show("v2_cat",n_iter=12,anchor_iters=0,skelpow=3.0,catconstrain=True)
    show("v2_anchor",n_iter=12,anchor_iters=6,skelpow=3.0,catconstrain=True)
    print("t=%.0fs"%(time.time()-t0))
