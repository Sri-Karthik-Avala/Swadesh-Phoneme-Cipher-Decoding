import numpy as np, collections, random, time
import exp, solve2, solve3, solve5, phon
from scipy.optimize import linear_sum_assignment

def run(target, n_iter=12, wpow=12.0, gap=-3.0, relpow=0.5, pin=50.0, vow_iters=6,
        vow_gap=-1.5, marg_alpha=0.5, final_polish=0):
    words=exp.by_lang[target]
    segset=sorted({s for _,segs in words for s in segs})
    rng=random.Random(12345); perm=segset[:]; rng.shuffle(perm)
    s2t={s:"x%d"%i for i,s in enumerate(perm)}
    tr=[(con,[s2t[s] for s in segs]) for con,segs in words]; true=[segs for con,segs in words]
    support,seg_pool=solve2.make_support(target)
    tok_vocab=sorted({t for _,toks in tr for t in toks}); tok_id={t:i for i,t in enumerate(tok_vocab)}
    seg_vocab=sorted(seg_pool); seg_id={s:i for i,s in enumerate(seg_vocab)}
    T=len(tok_vocab); S=len(seg_vocab)
    cf,co,rf,ro,pl,langs,li=exp.build_pairs(tr,support,seg_id,tok_id)
    nlang=len(langs); lang_w=np.ones(nlang)
    seg_isvow=np.array([1 if phon.feat_vec(s)[4]>0.5 else 0 for s in seg_vocab],dtype=np.int64)
    # phase1 EM
    C=np.ones((T,S))*0.01
    for p in range(co.shape[0]-1):
        ci=cf[co[p]:co[p+1]]; rr=rf[ro[p]:ro[p+1]]; w=1.0/(len(ci)*len(rr))
        for t in ci: C[t,rr]+=w
    sub=np.log(C/C.sum(1,keepdims=True)+1e-9)
    counts=None
    for it in range(n_iter):
        counts,lcov,lcnt=solve2.estep2(cf,co,rf,ro,pl,lang_w,sub,gap,T,S,nlang,relpow)
        cs=counts+1e-6; sub=np.log(cs/cs.sum(1,keepdims=True))
        cov=np.where(lcnt>0,lcov/np.maximum(lcnt,1),0.0); mx=cov.max(); lang_w=(cov/mx)**wpow
    vmass=(counts*seg_isvow[None,:]).sum(1)/(counts.sum(1)+1e-9)
    tok_isvow=(vmass>0.5).astype(np.int64)
    counts_cons=counts.copy()
    # phase2: positional vowel reading with hard-pinned consonants, uniform vowel sub
    for vit in range(vow_iters):
        # consonant map (within-consonant argmax)
        tokpred=np.zeros(T,dtype=np.int64)
        for t in range(T):
            row=counts[t].copy(); mask=(seg_isvow==tok_isvow[t])
            if mask.any(): row=np.where(mask,row,-1.0)
            tokpred[t]=int(np.argmax(row))
        sub_anchor=np.zeros((T,S))
        for t in range(T):
            if tok_isvow[t]==0:
                sub_anchor[t,:]=-pin
                sub_anchor[t,tokpred[t]]=pin
        vc,lcov,lcnt=solve5.estep_pin(cf,co,rf,ro,pl,lang_w,sub_anchor,vow_gap,T,S,nlang,tokpred,tok_isvow,0.0,0.0)
        # vc has counts for all; keep vowel rows from vc, consonant rows from EM
        counts=counts_cons.copy()
        for t in range(T):
            if tok_isvow[t]==1:
                counts[t]=vc[t]
        cov=np.where(lcnt>0,lcov/np.maximum(lcnt,1),0.0); mx=cov.max() if cov.max()>0 else 1; lang_w=(cov/mx)**wpow
    marg=counts.sum(0)+1e-9
    score=(counts+1e-9)/(marg[None,:]**marg_alpha)
    tm=solve3.category_assign(score,tok_isvow,seg_isvow,tok_vocab,seg_vocab)
    pred=[[tm.get(t,'?') for t in toks] for _,toks in tr]
    return exp.score_pred(pred,true)

if __name__=='__main__':
    SAAMI=['sme','smj','sjd','sms','smn','sma']; FINN=['fin','ekk','krl','olo','vep']
    t0=time.time()
    def show(name,**kw):
        sa=[run(tg,**kw) for tg in SAAMI]; fi=[run(tg,**kw) for tg in FINN]
        print("%-24s SAAMI=%.4f FINN=%.4f | %s"%(name,np.mean(sa),np.mean(fi),
              ' '.join('%s=%.3f'%(t,s) for t,s in zip(SAAMI,sa))))
    # baseline solve5 anchor0 for reference
    for vg in [-1.0,-1.5,-2.5]:
        show("posvow vg=%g pin50"%vg,vow_gap=vg,pin=50.0,vow_iters=6,marg_alpha=0.5)
    show("posvow vg-1.5 marg0",vow_gap=-1.5,pin=50.0,vow_iters=6,marg_alpha=0.0)
    show("posvow vg-1.5 marg0.5 it10",vow_gap=-1.5,pin=50.0,vow_iters=10,marg_alpha=0.5)
    print("t=%.0fs"%(time.time()-t0))
