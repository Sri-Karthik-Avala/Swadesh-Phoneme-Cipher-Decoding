import numpy as np, collections, random, time
import exp, solve2, solve3, solve5, phon
from scipy.optimize import linear_sum_assignment

def em(tr, support_langs, target, wpow=12,relpow=0.5,n_iter=12,anchor=0,skelpow=4.0,
       marg=0.0, cat=False, ret_cov=False):
    support=collections.defaultdict(list); seg_pool=set()
    for lg in support_langs:
        if lg==target: continue
        for con,segs in exp.by_lang[lg]:
            support[con].append((lg,segs)); seg_pool.update(segs)
    tok_vocab=sorted({t for _,toks in tr for t in toks}); tok_id={t:i for i,t in enumerate(tok_vocab)}
    seg_vocab=sorted(seg_pool); seg_id={s:i for i,s in enumerate(seg_vocab)}
    T=len(tok_vocab); S=len(seg_vocab)
    seg_isvow=np.array([1 if phon.feat_vec(s)[4]>0.5 else 0 for s in seg_vocab],dtype=np.int64)
    cf,co,rf,ro,pl,langs,li=exp.build_pairs(tr,support,seg_id,tok_id)
    nlang=len(langs); lang_w=np.ones(nlang)
    C=np.ones((T,S))*0.01
    for p in range(co.shape[0]-1):
        ci=cf[co[p]:co[p+1]]; rr=rf[ro[p]:ro[p+1]]; w=1.0/(len(ci)*len(rr))
        for t in ci: C[t,rr]+=w
    sub=np.log(C/C.sum(1,keepdims=True)+1e-9); counts=None; cov=None
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
        counts,lcov,lcnt=solve5.estep_pin(cf,co,rf,ro,pl,lang_w,sub,-3.0,T,S,nlang,tokpred,tok_isvow,skelpow,0.0)
        cs=counts+1e-6; sub=np.log(cs/cs.sum(1,keepdims=True))
        cov=np.where(lcnt>0,lcov/np.maximum(lcnt,1),0.0); lang_w=(cov/cov.max())**wpow
    m=counts.sum(0)+1e-12; score=(counts+1e-12)/(m[None,:]**marg)
    if cat: tm=solve3.category_assign(score,tok_isvow,seg_isvow,tok_vocab,seg_vocab)
    else:
        r,c=linear_sum_assignment(-score); tm={tok_vocab[ri]:seg_vocab[ci] for ri,ci in zip(r,c)}
    if ret_cov: return tm, np.sort(cov)[::-1][:3]
    return tm

def make_distant(target, remove_k):
    # remove target + its remove_k closest sisters (by a quick all-uralic cov) from support
    words=exp.by_lang[target]; segset=sorted({s for _,segs in words for s in segs})
    rng=random.Random(12345); perm=segset[:]; rng.shuffle(perm)
    s2t={s:"x%d"%i for i,s in enumerate(perm)}
    tr=[(con,[s2t[s] for s in segs]) for con,segs in words]; true=[segs for con,segs in words]
    # rank sisters
    support=collections.defaultdict(list); seg_pool=set()
    for lg in exp.URAL:
        if lg==target: continue
        for con,segs in exp.by_lang[lg]: support[con].append((lg,segs))
    _,_=None,None
    # quick cov via em ret_cov requires support_langs list; just run once
    tok_vocab=sorted({t for _,toks in tr for t in toks}); tok_id={t:i for i,t in enumerate(tok_vocab)}
    sp=set(s for lg in exp.URAL if lg!=target for _,segs in exp.by_lang[lg] for s in segs)
    seg_id={s:i for i,s in enumerate(sorted(sp))}
    cf,co,rf,ro,pl,langs,li=exp.build_pairs([('',c,toks) if False else (c,toks) for c,toks in tr],support,seg_id,tok_id)
    # simpler: use em ret_cov
    _,cov3=em(tr,exp.URAL,target,ret_cov=True)
    # need full cov per lang; recompute
    return tr,true

def rank_and_support(tr, target, remove_k):
    # get per-lang coverage under all-uralic, remove top-k closest
    support=collections.defaultdict(list)
    for lg in exp.URAL:
        if lg==target: continue
        for con,segs in exp.by_lang[lg]: support[con].append((lg,segs))
    tok_vocab=sorted({t for _,toks in tr for t in toks}); tok_id={t:i for i,t in enumerate(tok_vocab)}
    sp=sorted(set(s for lg in exp.URAL if lg!=target for _,segs in exp.by_lang[lg] for s in segs))
    seg_id={s:i for i,s in enumerate(sp)}; T=len(tok_vocab); S=len(sp)
    cf,co,rf,ro,pl,langs,li=exp.build_pairs(tr,support,seg_id,tok_id)
    nlang=len(langs); lang_w=np.ones(nlang)
    C=np.ones((T,S))*0.01
    for p in range(co.shape[0]-1):
        ci=cf[co[p]:co[p+1]]; rr=rf[ro[p]:ro[p+1]]; w=1.0/(len(ci)*len(rr))
        for t in ci: C[t,rr]+=w
    sub=np.log(C/C.sum(1,keepdims=True)+1e-9)
    for it in range(8):
        counts,lcov,lcnt=solve2.estep2(cf,co,rf,ro,pl,lang_w,sub,-3.0,T,S,nlang,0.5)
        cs=counts+1e-6; sub=np.log(cs/cs.sum(1,keepdims=True))
        cov=np.where(lcnt>0,lcov/np.maximum(lcnt,1),0.0); lang_w=(cov/cov.max())**12
    ranked=[langs[i] for i in np.argsort(-cov)]
    remove=set([target]+ranked[:remove_k])
    keep=[lg for lg in exp.URAL if lg not in remove]
    return keep

if __name__=='__main__':
    t0=time.time()
    FINN=['fin','ekk','krl','olo','vep']
    for rk in [0,2,3]:
        print("=== remove top-%d closest sisters (simulate distant target) ==="%rk)
        for tg in FINN:
            words=exp.by_lang[tg]; segset=sorted({s for _,segs in words for s in segs})
            rng=random.Random(12345); perm=segset[:]; rng.shuffle(perm)
            s2t={s:"x%d"%i for i,s in enumerate(perm)}
            tr=[(con,[s2t[s] for s in segs]) for con,segs in words]; true=[segs for con,segs in words]
            keep=rank_and_support(tr,tg,rk)
            def sc(tm): return exp.score_pred([[tm.get(t,'?') for t in toks] for _,toks in tr],true)
            plain=sc(em(tr,keep,tg))
            _,cov=em(tr,keep,tg,ret_cov=True)
            cat=sc(em(tr,keep,tg,cat=True))
            marg3=sc(em(tr,keep,tg,cat=True,marg=0.3))
            marg5=sc(em(tr,keep,tg,cat=True,marg=0.5))
            anch=sc(em(tr,keep,tg,cat=True,anchor=6))
            print("  %s topcov=%.3f | plain=%.3f cat=%.3f marg.3=%.3f marg.5=%.3f anchor=%.3f"%(
                tg,cov[0],plain,cat,marg3,marg5,anch))
    print("t=%.0fs"%(time.time()-t0))
