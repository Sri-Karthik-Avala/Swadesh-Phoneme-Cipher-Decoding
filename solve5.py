import csv, collections, random, time
import numpy as np
from numba import njit
from scipy.optimize import linear_sum_assignment
import exp, solve2, solve3, phon

@njit(cache=True)
def estep_pin(cip_flat,cip_off,ref_flat,ref_off,pair_lang,lang_w,sub_score,gap,T,S,nlang,
              tokpred,tok_isvow,skelpow,consbonus):
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
            ti=cip_flat[cs+i-1]; pin=tokpred[ti]; isc=(tok_isvow[ti]==0)
            for j in range(1,m+1):
                sj=ref_flat[rs+j-1]
                sc=sub_score[ti,sj]
                if isc and sj==pin: sc+=consbonus
                diag=D[i-1,j-1]+sc
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
            if tok_isvow[t]==0 and ms[a]==tokpred[t]: cmatch+=1
        denom=ncons_word if ncons_word>0 else 1
        skel=cmatch/denom
        pw=w*(skel**skelpow)
        for a in range(matched):
            counts[mt[a],ms[a]]+=pw
        mx=n if n>m else m
        lang_cov[lg]+=matched/mx; lang_cnt[lg]+=1.0
    return counts,lang_cov,lang_cnt

def decipher5(test_rows, support_by_concept, seg_pool, n_iter=12, wpow=12.0, gap=-3.0,
              relpow=0.5, anchor_iters=8, skelpow=4.0, consbonus=6.0, marg_alpha=0.5,
              vow_minfreq=0, catconstrain=True):
    tok_vocab=sorted({t for _,toks in test_rows for t in toks})
    tok_id={t:i for i,t in enumerate(tok_vocab)}
    seg_vocab=sorted(seg_pool); seg_id={s:i for i,s in enumerate(seg_vocab)}
    T=len(tok_vocab); S=len(seg_vocab)
    cf,co,rf,ro,pl,langs,li=exp.build_pairs(test_rows,support_by_concept,seg_id,tok_id)
    nlang=len(langs); lang_w=np.ones(nlang)
    seg_isvow=np.array([1 if phon.feat_vec(s)[4]>0.5 else 0 for s in seg_vocab],dtype=np.int64)
    # seg corpus freq (for candidate restriction)
    segfreq=np.zeros(S)
    for lg,rows in ((l,exp.by_lang[l]) for l in support_by_concept and []):
        pass
    C=np.ones((T,S))*0.01
    for p in range(co.shape[0]-1):
        ci=cf[co[p]:co[p+1]]; rr=rf[ro[p]:ro[p+1]]; w=1.0/(len(ci)*len(rr))
        for t in ci: C[t,rr]+=w
        for s in rr: segfreq[s]+=1
    sub=np.log(C/C.sum(1,keepdims=True)+1e-9)
    counts=None
    for it in range(n_iter):
        counts,lcov,lcnt=solve2.estep2(cf,co,rf,ro,pl,lang_w,sub,gap,T,S,nlang,relpow)
        cs=counts+1e-6; sub=np.log(cs/cs.sum(1,keepdims=True))
        cov=np.where(lcnt>0,lcov/np.maximum(lcnt,1),0.0); mx=cov.max()
        if mx>0: lang_w=(cov/mx)**wpow
    vmass=(counts*seg_isvow[None,:]).sum(1)/(counts.sum(1)+1e-9)
    tok_isvow=(vmass>0.5).astype(np.int64)
    for it in range(anchor_iters):
        tokpred=np.zeros(T,dtype=np.int64)
        for t in range(T):
            row=counts[t].copy(); mask=(seg_isvow==tok_isvow[t])
            if mask.any(): row=np.where(mask,row,-1.0)
            tokpred[t]=int(np.argmax(row))
        counts,lcov,lcnt=estep_pin(cf,co,rf,ro,pl,lang_w,sub,gap,T,S,nlang,tokpred,tok_isvow,skelpow,consbonus)
        cs=counts+1e-6; sub=np.log(cs/cs.sum(1,keepdims=True))
        cov=np.where(lcnt>0,lcov/np.maximum(lcnt,1),0.0); mx=cov.max()
        if mx>0: lang_w=(cov/mx)**wpow
    marg=counts.sum(0)+1e-9
    score=(counts+1e-9)/(marg[None,:]**marg_alpha)
    # candidate restriction: kill segs below freq threshold (keep them only if needed)
    if vow_minfreq>0:
        rare=segfreq<vow_minfreq
        score[:,rare]*=1e-6
    if catconstrain:
        return solve3.category_assign(score,tok_isvow,seg_isvow,tok_vocab,seg_vocab)
    r,c=linear_sum_assignment(-score)
    return {tok_vocab[ri]:seg_vocab[ci] for ri,ci in zip(r,c)}

def run5(target, **kw):
    words=exp.by_lang[target]
    segset=sorted({s for _,segs in words for s in segs})
    rng=random.Random(12345); perm=segset[:]; rng.shuffle(perm)
    s2t={s:"x%d"%i for i,s in enumerate(perm)}
    tr=[(con,[s2t[s] for s in segs]) for con,segs in words]; true=[segs for con,segs in words]
    support,seg_pool=solve2.make_support(target)
    tm=decipher5(tr,support,seg_pool,**kw)
    pred=[[tm.get(t,'?') for t in toks] for _,toks in tr]
    return exp.score_pred(pred,true)

if __name__=='__main__':
    SAAMI=['sme','smj','sjd','sms','smn','sma']
    FINN=['fin','ekk','krl','olo','vep']
    t0=time.time()
    def show(name,**kw):
        sa=[run5(tg,**kw) for tg in SAAMI]
        fi=[run5(tg,**kw) for tg in FINN]
        print("%-28s SAAMI=%.4f FINN=%.4f | saami:%s"%(name,np.mean(sa),np.mean(fi),
              ' '.join('%s=%.3f'%(t,s) for t,s in zip(SAAMI,sa))))
    show("baseline(anc0)",anchor_iters=0,marg_alpha=0.0,catconstrain=False,consbonus=0.0,skelpow=4.0)
    show("pin cb6 sp4 m0.5",anchor_iters=8,consbonus=6.0,skelpow=4.0,marg_alpha=0.5)
    show("pin cb10 sp4 m0.5",anchor_iters=8,consbonus=10.0,skelpow=4.0,marg_alpha=0.5)
    show("pin cb10 sp6 m0.5",anchor_iters=8,consbonus=10.0,skelpow=6.0,marg_alpha=0.5)
    show("pin cb10 sp4 m0.3",anchor_iters=8,consbonus=10.0,skelpow=4.0,marg_alpha=0.3)
    print("t=%.0fs"%(time.time()-t0))
