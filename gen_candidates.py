import csv, collections, sys
import numpy as np
import exp, solve2, solve3, solve5, phon
from scipy.optimize import linear_sum_assignment

def load_real():
    rows=[]
    with open('test.csv',encoding='utf-8') as f:
        for row in csv.DictReader(f): rows.append((row['id'],row['concept'],row['cipher'].split()))
    return rows

def base_em(tr, support, seg_pool, wpow=12.0, relpow=0.5, gap=-3.0, n_iter=12,
            anchor_iters=0, skelpow=4.0, consbonus=0.0):
    tok_vocab=sorted({t for _,toks in tr for t in toks}); tok_id={t:i for i,t in enumerate(tok_vocab)}
    seg_vocab=sorted(seg_pool); seg_id={s:i for i,s in enumerate(seg_vocab)}
    T=len(tok_vocab); S=len(seg_vocab)
    cf,co,rf,ro,pl,langs,li=exp.build_pairs(tr,support,seg_id,tok_id)
    nlang=len(langs); lang_w=np.ones(nlang)
    seg_isvow=np.array([1 if phon.feat_vec(s)[4]>0.5 else 0 for s in seg_vocab],dtype=np.int64)
    C=np.ones((T,S))*0.01
    for p in range(co.shape[0]-1):
        ci=cf[co[p]:co[p+1]]; rr=rf[ro[p]:ro[p+1]]; w=1.0/(len(ci)*len(rr))
        for t in ci: C[t,rr]+=w
    sub=np.log(C/C.sum(1,keepdims=True)+1e-9); counts=None
    for it in range(n_iter):
        counts,lcov,lcnt=solve2.estep2(cf,co,rf,ro,pl,lang_w,sub,gap,T,S,nlang,relpow)
        cs=counts+1e-6; sub=np.log(cs/cs.sum(1,keepdims=True))
        cov=np.where(lcnt>0,lcov/np.maximum(lcnt,1),0.0); lang_w=(cov/cov.max())**wpow
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
        cov=np.where(lcnt>0,lcov/np.maximum(lcnt,1),0.0); lang_w=(cov/cov.max())**wpow
    return counts, tok_vocab, seg_vocab, tok_isvow, seg_isvow

def assign(counts, tok_vocab, seg_vocab, tok_isvow, seg_isvow, marg=0.0, cat=True):
    m=counts.sum(0)+1e-12
    score=(counts+1e-12)/(m[None,:]**marg)
    if cat:
        return solve3.category_assign(score,tok_isvow,seg_isvow,tok_vocab,seg_vocab)
    r,c=linear_sum_assignment(-score)
    return {tok_vocab[ri]:seg_vocab[ci] for ri,ci in zip(r,c)}

def write_sub(path, test_rows, tm):
    with open(path,'w',encoding='utf-8',newline='') as f:
        w=csv.writer(f); w.writerow(['id','ipa'])
        for _id,con,toks in test_rows:
            w.writerow([_id,' '.join(tm.get(t,'a') for t in toks)])
    print("wrote",path)

if __name__=='__main__':
    test_rows=load_real(); tr=[(c,toks) for _id,c,toks in test_rows]
    support=collections.defaultdict(list); seg_pool=set()
    for lg in exp.URAL:
        for con,segs in exp.by_lang[lg]:
            support[con].append((lg,segs)); seg_pool.update(segs)
    # base EM (no anchor)
    counts,tv,sv,tiv,siv=base_em(tr,support,seg_pool,anchor_iters=0)
    write_sub('working/cand_A_plain.csv',test_rows,assign(counts,tv,sv,tiv,siv,marg=0.0,cat=False))
    write_sub('working/cand_B_cat.csv',test_rows,assign(counts,tv,sv,tiv,siv,marg=0.0,cat=True))
    write_sub('working/cand_C_marg03.csv',test_rows,assign(counts,tv,sv,tiv,siv,marg=0.3,cat=True))
    write_sub('working/cand_D_marg05.csv',test_rows,assign(counts,tv,sv,tiv,siv,marg=0.5,cat=True))
    # ensemble of wpow, raw-count average
    acc=np.zeros_like(counts)
    for wp,rp in [(8.0,0.3),(12.0,0.5),(16.0,0.8)]:
        cc,tv2,sv2,tiv2,siv2=base_em(tr,support,seg_pool,wpow=wp,relpow=rp,anchor_iters=0)
        acc+=cc/cc.sum()
    write_sub('working/cand_E_ensemble.csv',test_rows,assign(acc,tv2,sv2,tiv2,siv2,marg=0.0,cat=True))
    # anchored
    counts2,tv3,sv3,tiv3,siv3=base_em(tr,support,seg_pool,anchor_iters=6,skelpow=4.0)
    write_sub('working/cand_F_anchor.csv',test_rows,assign(counts2,tv3,sv3,tiv3,siv3,marg=0.0,cat=True))
