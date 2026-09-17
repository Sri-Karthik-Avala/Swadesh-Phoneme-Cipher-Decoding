import csv, collections, random, time
import numpy as np
from scipy.optimize import linear_sum_assignment
import exp, solve2, solve3, phon

def decipher6(test_rows, support_by_concept, seg_pool, n_iter=12, wpow=12.0, gap=-3.0,
              relpow=0.5, marg_alpha=0.8, nvow=40, ncons=55, catconstrain=True):
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
    for it in range(n_iter):
        counts,lcov,lcnt=solve2.estep2(cf,co,rf,ro,pl,lang_w,sub,gap,T,S,nlang,relpow)
        cs=counts+1e-6; sub=np.log(cs/cs.sum(1,keepdims=True))
        cov=np.where(lcnt>0,lcov/np.maximum(lcnt,1),0.0); mx=cov.max()
        if mx>0: lang_w=(cov/mx)**wpow
    vmass=(counts*seg_isvow[None,:]).sum(1)/(counts.sum(1)+1e-9)
    tok_isvow=(vmass>0.5).astype(np.int64)
    # segment "importance" = total counts mass (how much target actually uses it)
    segmass=counts.sum(0)
    # restrict candidate pool per category to top-N by mass
    keep=np.zeros(S,dtype=bool)
    vidx=np.where(seg_isvow==1)[0]; cidx=np.where(seg_isvow==0)[0]
    vtop=vidx[np.argsort(-segmass[vidx])[:nvow]]
    ctop=cidx[np.argsort(-segmass[cidx])[:ncons]]
    keep[vtop]=True; keep[ctop]=True
    marg=segmass+1e-9
    score=(counts+1e-9)/(marg[None,:]**marg_alpha)
    score[:,~keep]*=1e-9
    if catconstrain:
        return solve3.category_assign(score,tok_isvow,seg_isvow,tok_vocab,seg_vocab), (counts,tok_vocab,tok_id,seg_vocab,tok_isvow,seg_isvow)
    r,c=linear_sum_assignment(-score)
    return {tok_vocab[ri]:seg_vocab[ci] for ri,ci in zip(r,c)}, None

def run6(target, **kw):
    words=exp.by_lang[target]
    segset=sorted({s for _,segs in words for s in segs})
    rng=random.Random(12345); perm=segset[:]; rng.shuffle(perm)
    s2t={s:"x%d"%i for i,s in enumerate(perm)}
    tr=[(con,[s2t[s] for s in segs]) for con,segs in words]; true=[segs for con,segs in words]
    support,seg_pool=solve2.make_support(target)
    tm,_=decipher6(tr,support,seg_pool,**kw)
    pred=[[tm.get(t,'?') for t in toks] for _,toks in tr]
    return exp.score_pred(pred,true)

if __name__=='__main__':
    FINN=['fin','ekk','krl','olo','vep']; SAAMI=['sme','smj','sjd','sms']
    t0=time.time()
    def show(name,**kw):
        fi=[run6(tg,**kw) for tg in FINN]; sa=[run6(tg,**kw) for tg in SAAMI]
        print("%-26s ekk=%.3f FINN=%.4f SAAMI=%.4f | %s"%(name,fi[1],np.mean(fi),np.mean(sa),
              ' '.join('%s=%.3f'%(t,s) for t,s in zip(FINN,fi))))
    show("marg0 (ref)",marg_alpha=0.0,nvow=999,ncons=999)
    for ma in [0.5,0.8,1.0]:
        for nv in [30,45]:
            show("marg%.1f nv%d nc60"%(ma,nv),marg_alpha=ma,nvow=nv,ncons=60)
    print("t=%.0fs"%(time.time()-t0))
