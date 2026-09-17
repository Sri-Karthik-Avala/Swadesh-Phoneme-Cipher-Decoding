import numpy as np, collections, random, time
import exp, solve2
from scipy.optimize import linear_sum_assignment

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

def selfconsist(tm, tr, support):
    # mean over words of max over sisters of segment similarity(decoded, sister)
    tot=0.0; nz=0
    for con,toks in tr:
        dec=[tm.get(t,'?') for t in toks]
        best=0.0
        for lg,segs in support.get(con,[]):
            m=max(len(dec),len(segs))
            if m==0: continue
            sim=1-lev(dec,segs)/m
            if sim>best: best=sim
        tot+=best; nz+=1
    return tot/nz

def run_em(tr, support, seg_pool, init='cooc', seed=0, n_iter=12, wpow=12.0, gap=-3.0, relpow=0.5):
    tok_vocab=sorted({t for _,toks in tr for t in toks}); tok_id={t:i for i,t in enumerate(tok_vocab)}
    seg_vocab=sorted(seg_pool); seg_id={s:i for i,s in enumerate(seg_vocab)}
    T=len(tok_vocab); S=len(seg_vocab)
    cf,co,rf,ro,pl,langs,li=exp.build_pairs(tr,support,seg_id,tok_id)
    nlang=len(langs); lang_w=np.ones(nlang)
    if init=='cooc':
        C=np.ones((T,S))*0.01
        for p in range(co.shape[0]-1):
            ci=cf[co[p]:co[p+1]]; rr=rf[ro[p]:ro[p+1]]; w=1.0/(len(ci)*len(rr))
            for t in ci: C[t,rr]+=w
        sub=np.log(C/C.sum(1,keepdims=True)+1e-9)
    else:  # random
        rng=np.random.RandomState(seed)
        sub=rng.rand(T,S)*2.0-1.0
    counts=None
    for it in range(n_iter):
        counts,lcov,lcnt=solve2.estep2(cf,co,rf,ro,pl,lang_w,sub,gap,T,S,nlang,relpow)
        cs=counts+1e-6; sub=np.log(cs/cs.sum(1,keepdims=True))
        cov=np.where(lcnt>0,lcov/np.maximum(lcnt,1),0.0); mx=cov.max(); lang_w=(cov/mx)**wpow
    r,c=linear_sum_assignment(-(counts+1e-9))
    tm={tok_vocab[ri]:seg_vocab[ci] for ri,ci in zip(r,c)}
    return tm

def eval_target(target, n_restarts=8):
    words=exp.by_lang[target]
    segset=sorted({s for _,segs in words for s in segs})
    rng=random.Random(12345); perm=segset[:]; rng.shuffle(perm)
    s2t={s:"x%d"%i for i,s in enumerate(perm)}
    tr=[(con,[s2t[s] for s in segs]) for con,segs in words]; true=[segs for con,segs in words]
    support,seg_pool=solve2.make_support(target)
    def sc(tm):
        pred=[[tm.get(t,'?') for t in toks] for _,toks in tr]
        return exp.score_pred(pred,true)
    cands=[('cooc',run_em(tr,support,seg_pool,init='cooc'))]
    for s in range(n_restarts):
        cands.append(('rand%d'%s, run_em(tr,support,seg_pool,init='random',seed=s)))
    rows=[]
    for name,tm in cands:
        rows.append((name, sc(tm), selfconsist(tm,tr,support)))
    # selection by selfconsist
    best=max(rows,key=lambda r:r[2])
    truebest=max(rows,key=lambda r:r[1])
    return rows, best, truebest

if __name__=='__main__':
    t0=time.time()
    for tg in ['sme','sjd','ekk','smj']:
        rows,best,truebest=eval_target(tg,n_restarts=8)
        cooc=[r for r in rows if r[0]=='cooc'][0]
        print("%s: cooc(sc=%.3f,cons=%.3f)  SELECTED-by-cons=%s(sc=%.3f)  best-possible=%.3f"%(
            tg,cooc[1],cooc[2],best[0],best[1],truebest[1]))
        sr=sorted(rows,key=lambda r:-r[2])[:4]
        print("   top-by-consist:",[(n,round(s,3),round(c,3)) for n,s,c in sr])
    print("t=%.0fs"%(time.time()-t0))
