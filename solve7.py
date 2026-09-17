import csv, collections, random, time
import numpy as np
from scipy.optimize import linear_sum_assignment
import exp, solve2, solve3, solve5, phon

def one_config(cf,co,rf,ro,pl,langs,T,S,seg_isvow,n_iter,wpow,gap,relpow,anchor_iters,skelpow,consbonus):
    nlang=len(langs); lang_w=np.ones(nlang)
    C=np.ones((T,S))*0.01
    for p in range(co.shape[0]-1):
        ci=cf[co[p]:co[p+1]]; rr=rf[ro[p]:ro[p+1]]; w=1.0/(len(ci)*len(rr))
        for t in ci: C[t,rr]+=w
    sub=np.log(C/C.sum(1,keepdims=True)+1e-9)
    counts=None; cov=None
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
        counts,lcov,lcnt=solve5.estep_pin(cf,co,rf,ro,pl,lang_w,sub,gap,T,S,nlang,tokpred,tok_isvow,skelpow,consbonus)
        cs=counts+1e-6; sub=np.log(cs/cs.sum(1,keepdims=True))
        cov=np.where(lcnt>0,lcov/np.maximum(lcnt,1),0.0); mx=cov.max()
        if mx>0: lang_w=(cov/mx)**wpow
    q=np.sort(cov)[::-1][:5].mean()
    cn=counts/(counts.sum(1,keepdims=True)+1e-9)
    return cn, counts, q, tok_isvow

def decipher7(test_rows, support_by_concept, seg_pool, configs, marg_alpha=0.5, nvow=40, ncons=60,
              return_meta=False):
    tok_vocab=sorted({t for _,toks in test_rows for t in toks}); tok_id={t:i for i,t in enumerate(tok_vocab)}
    seg_vocab=sorted(seg_pool); seg_id={s:i for i,s in enumerate(seg_vocab)}
    T=len(tok_vocab); S=len(seg_vocab)
    cf,co,rf,ro,pl,langs,li=exp.build_pairs(test_rows,support_by_concept,seg_id,tok_id)
    seg_isvow=np.array([1 if phon.feat_vec(s)[4]>0.5 else 0 for s in seg_vocab],dtype=np.int64)
    ens=np.zeros((T,S)); rawsum=np.zeros((T,S)); tv_acc=np.zeros(T)
    for cfg in configs:
        cn,counts,q,tok_isvow=one_config(cf,co,rf,ro,pl,langs,T,S,seg_isvow,**cfg)
        ens+=cn*q; rawsum+=counts*q; tv_acc+=tok_isvow*q
    tok_isvow=(tv_acc>0.5*sum(1 for _ in configs)*np.mean([1])).astype(np.int64)
    # recompute tok category by ensemble vote weighted
    tot_q=sum(c for c in [1]*len(configs))
    tok_isvow=((ens*seg_isvow[None,:]).sum(1)/(ens.sum(1)+1e-9)>0.5).astype(np.int64)
    segmass=rawsum.sum(0)
    keep=np.zeros(S,dtype=bool)
    vidx=np.where(seg_isvow==1)[0]; cidx=np.where(seg_isvow==0)[0]
    keep[vidx[np.argsort(-segmass[vidx])[:nvow]]]=True
    keep[cidx[np.argsort(-segmass[cidx])[:ncons]]]=True
    marg=segmass+1e-9
    score=(ens+1e-12)/(marg[None,:]**marg_alpha)
    score[:,~keep]*=1e-9
    tm=solve3.category_assign(score,tok_isvow,seg_isvow,tok_vocab,seg_vocab)
    if return_meta:
        return tm,(ens,rawsum,tok_vocab,tok_id,seg_vocab,tok_isvow,seg_isvow)
    return tm

BASE=dict(n_iter=12,gap=-3.0,skelpow=4.0,consbonus=0.0)
ENSEMBLE=[
  dict(wpow=8.0,relpow=0.3,anchor_iters=0,**BASE),
  dict(wpow=12.0,relpow=0.5,anchor_iters=0,**BASE),
  dict(wpow=16.0,relpow=0.8,anchor_iters=0,**BASE),
  dict(wpow=12.0,relpow=0.5,anchor_iters=6,**BASE),
  dict(wpow=10.0,relpow=0.5,anchor_iters=6,skelpow=4.0,consbonus=8.0,n_iter=12,gap=-3.0),
]
SINGLE=[dict(wpow=12.0,relpow=0.5,anchor_iters=0,**BASE)]

def run7(target, configs, **kw):
    words=exp.by_lang[target]
    segset=sorted({s for _,segs in words for s in segs})
    rng=random.Random(12345); perm=segset[:]; rng.shuffle(perm)
    s2t={s:"x%d"%i for i,s in enumerate(perm)}
    tr=[(con,[s2t[s] for s in segs]) for con,segs in words]; true=[segs for con,segs in words]
    support,seg_pool=solve2.make_support(target)
    tm=decipher7(tr,support,seg_pool,configs,**kw)
    pred=[[tm.get(t,'?') for t in toks] for _,toks in tr]
    return exp.score_pred(pred,true)

def run_plain(target):
    words=exp.by_lang[target]
    segset=sorted({s for _,segs in words for s in segs})
    rng=random.Random(12345); perm=segset[:]; rng.shuffle(perm)
    s2t={s:"x%d"%i for i,s in enumerate(perm)}
    tr=[(con,[s2t[s] for s in segs]) for con,segs in words]; true=[segs for con,segs in words]
    support,seg_pool=solve2.make_support(target)
    tm=solve2.decipher2(tr,support,seg_pool,n_iter=12,wpow=12.0,gap=-3.0,relpow=0.5,smooth=False)
    pred=[[tm.get(t,'?') for t in toks] for _,toks in tr]
    return exp.score_pred(pred,true)

if __name__=='__main__':
    FINN=['fin','ekk','krl','olo','vep']; SAAMI=['sme','smj','sjd','sms']
    t0=time.time()
    pl_fi=[run_plain(tg) for tg in FINN]; pl_sa=[run_plain(tg) for tg in SAAMI]
    print("%-24s ekk=%.3f FINN=%.4f SAAMI=%.4f all=%.4f | %s"%('PLAIN-v1',pl_fi[1],np.mean(pl_fi),np.mean(pl_sa),
          np.mean(pl_fi+pl_sa),' '.join('%s=%.3f'%(t,s) for t,s in zip(FINN+SAAMI,pl_fi+pl_sa))))
    for name,cfgs,ma,nv,nc in [('single+marg0.5',SINGLE,0.5,999,999),
                               ('ENSEMBLE marg0.5',ENSEMBLE,0.5,999,999),
                               ('ENSEMBLE marg0.5 restr',ENSEMBLE,0.5,40,60)]:
        fi=[run7(tg,cfgs,marg_alpha=ma,nvow=nv,ncons=nc) for tg in FINN]
        sa=[run7(tg,cfgs,marg_alpha=ma,nvow=nv,ncons=nc) for tg in SAAMI]
        print("%-24s ekk=%.3f FINN=%.4f SAAMI=%.4f all=%.4f | %s"%(name,fi[1],np.mean(fi),np.mean(sa),
              np.mean(fi+sa),' '.join('%s=%.3f'%(t,s) for t,s in zip(FINN+SAAMI,fi+sa))))
    print("t=%.0fs"%(time.time()-t0))
