import numpy as np, collections, random, time
from numba import njit
from scipy.optimize import linear_sum_assignment
import exp, solve2, solve3, phon

def base_class(seg):
    b = seg
    for d in ["ː","ˑ","ʲ","ʷ","ʰ","̥","̊","ʼ","́","̀","̄","̂","̃","̈","̪","̻","̠","ⁿ","ʲ"]:
        b = b.replace(d,"")
    return b if b else seg

@njit(cache=True)
def estep_pin_coarse(cip_flat,cip_off,ref_flat,ref_off,pair_lang,lang_w,sub_score,gap,T,S,nlang,
                     tokpred_base,tok_isvow,seg_base,skelpow):
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
        ncons_word=0
        for i2 in range(n):
            if tok_isvow[cip_flat[cs+i2]]==0: ncons_word+=1
        cmatch=0
        for a in range(matched):
            t=mt[a]
            if tok_isvow[t]==0 and seg_base[ms[a]]==tokpred_base[t]: cmatch+=1
        denom=ncons_word if ncons_word>0 else 1
        skel=cmatch/denom
        pw=w*(skel**skelpow)
        for a in range(matched):
            counts[mt[a],ms[a]]+=pw
        mx=n if n>m else m
        lang_cov[lg]+=matched/mx; lang_cnt[lg]+=1.0
    return counts,lang_cov,lang_cnt

def decipher_v3(tr, support_langs, target, wpow=12,relpow=0.5,n_iter=12,anchor=6,skelpow=6.0,
                coarse=True, cat=True, ret_tm=True):
    support=collections.defaultdict(list); seg_pool=set()
    for lg in support_langs:
        if lg==target: continue
        for con,segs in exp.by_lang[lg]:
            support[con].append((lg,segs)); seg_pool.update(segs)
    tok_vocab=sorted({t for _,toks in tr for t in toks}); tok_id={t:i for i,t in enumerate(tok_vocab)}
    seg_vocab=sorted(seg_pool); seg_id={s:i for i,s in enumerate(seg_vocab)}
    T=len(tok_vocab); S=len(seg_vocab)
    seg_isvow=np.array([1 if phon.feat_vec(s)[4]>0.5 else 0 for s in seg_vocab],dtype=np.int64)
    # base class ids
    bases=sorted(set(base_class(s) for s in seg_vocab)); base_id={b:i for i,b in enumerate(bases)}
    seg_base=np.array([base_id[base_class(s)] for s in seg_vocab],dtype=np.int64)
    cf,co,rf,ro,pl,langs,li=exp.build_pairs(tr,support,seg_id,tok_id)
    nlang=len(langs); lang_w=np.ones(nlang)
    C=np.ones((T,S))*0.01
    for p in range(co.shape[0]-1):
        ci=cf[co[p]:co[p+1]]; rr=rf[ro[p]:ro[p+1]]; w=1.0/(len(ci)*len(rr))
        for t in ci: C[t,rr]+=w
    sub=np.log(C/C.sum(1,keepdims=True)+1e-9); counts=None
    for it in range(n_iter):
        counts,lcov,lcnt=solve2.estep2(cf,co,rf,ro,pl,lang_w,sub,-3.0,T,S,nlang,relpow)
        cs=counts+1e-6; sub=np.log(cs/cs.sum(1,keepdims=True))
        cov=np.where(lcnt>0,lcov/np.maximum(lcnt,1),0.0); lang_w=(cov/cov.max())**wpow
    tok_isvow=((counts*seg_isvow[None,:]).sum(1)/(counts.sum(1)+1e-9)>0.5).astype(np.int64)
    for it in range(anchor):
        tokpred=np.zeros(T,dtype=np.int64)
        for t in range(T):
            row=counts[t].copy(); mask=(seg_isvow==tok_isvow[t])
            if mask.any(): row=np.where(mask,row,-1.0)
            tokpred[t]=int(np.argmax(row))
        tokpred_base=np.array([seg_base[tokpred[t]] for t in range(T)],dtype=np.int64)
        if coarse:
            counts,lcov,lcnt=estep_pin_coarse(cf,co,rf,ro,pl,lang_w,sub,-3.0,T,S,nlang,tokpred_base,tok_isvow,seg_base,skelpow)
        else:
            import solve5
            counts,lcov,lcnt=solve5.estep_pin(cf,co,rf,ro,pl,lang_w,sub,-3.0,T,S,nlang,tokpred,tok_isvow,skelpow,0.0)
        cs=counts+1e-6; sub=np.log(cs/cs.sum(1,keepdims=True))
        cov=np.where(lcnt>0,lcov/np.maximum(lcnt,1),0.0); lang_w=(cov/cov.max())**wpow
    score=counts+1e-12
    if cat: tm=solve3.category_assign(score,tok_isvow,seg_isvow,tok_vocab,seg_vocab)
    else:
        r,c=linear_sum_assignment(-score); tm={tok_vocab[ri]:seg_vocab[ci] for ri,ci in zip(r,c)}
    return tm

import distant
def prep(tg):
    words=exp.by_lang[tg]; segset=sorted({s for _,segs in words for s in segs})
    rng=random.Random(12345); perm=segset[:]; rng.shuffle(perm)
    s2t={s:"x%d"%i for i,s in enumerate(perm)}
    return [(con,[s2t[s] for s in segs]) for con,segs in words],[segs for con,segs in words]
def sc(tm,tr,true): return exp.score_pred([[tm.get(t,'?') for t in toks] for _,toks in tr],true)

if __name__=='__main__':
    t0=time.time()
    # moderate-Finnic analogs (real plain~0.57 matches these) + fine-transcribed Saami
    tests=[('ekk',2),('vep',3),('krl',3),('olo',3),('sme',2),('smn',3),('sjd',2),('sms',3)]
    print("target rk | plain  anchorFINE anchorCOARSE")
    for tg,rk in tests:
        tr,true=prep(tg); keep=distant.rank_and_support(tr,tg,rk)
        p=sc(distant.em(tr,keep,tg),tr,true)
        af=sc(decipher_v3(tr,keep,tg,anchor=6,skelpow=6.0,coarse=False),tr,true)
        ac=sc(decipher_v3(tr,keep,tg,anchor=6,skelpow=6.0,coarse=True),tr,true)
        print("  %-4s %d | %.3f  %.3f     %.3f  (%+.3f vs plain)"%(tg,rk,p,af,ac,ac-p))
    print("t=%.0fs"%(time.time()-t0))
