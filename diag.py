import collections, random, numpy as np, sys
import exp, solve2

def diagnose(target='ekk', **kw):
    words=exp.by_lang[target]
    segset=sorted({s for _,segs in words for s in segs})
    rng=random.Random(12345); perm=segset[:]; rng.shuffle(perm)
    seg2tok={s:"x%d"%i for i,s in enumerate(perm)}
    tok2seg_true={v:k for k,v in seg2tok.items()}
    test_rows=[]; true_words=[]
    for con,segs in words:
        test_rows.append((con,[seg2tok[s] for s in segs])); true_words.append(segs)
    support,seg_pool=solve2.make_support(target)
    tokmap,tok_vocab,tok_id,seg_vocab,counts=solve2.decipher2(
        test_rows,support,seg_pool,return_counts=True,**kw)
    # token frequency
    tokfreq=collections.Counter(t for _,toks in test_rows for t in toks)
    # per-token: true seg, predicted seg, correct?, and whether true seg is even in pool
    tot=sum(tokfreq.values())
    err_mass=0; oov_mass=0
    print("token  freq  true -> pred   [OK/ERR]  true_in_pool?")
    rows=[]
    for t,fr in tokfreq.most_common():
        true=tok2seg_true[t]; pred=tokmap.get(t,'?')
        inpool = true in set(seg_vocab)
        ok = (true==pred)
        if not ok: err_mass+=fr
        if not inpool: oov_mass+=fr
        rows.append((t,fr,true,pred,ok,inpool))
    for t,fr,true,pred,ok,inpool in rows[:45]:
        flag='OK ' if ok else 'ERR'
        print("%-5s %4d  %-6s -> %-6s  %s  %s"%(t,fr,true,pred,flag,'' if inpool else 'OOV'))
    print("\nTotal seg-occurrences:",tot)
    print("Error mass: %d (%.3f)  -> segacc=%.3f"%(err_mass,err_mass/tot,1-err_mass/tot))
    print("OOV mass (true seg not in support pool, unrecoverable): %d (%.3f)"%(oov_mass,oov_mass/tot))
    print("Recoverable error mass (err but in pool): %.3f"%((err_mass-oov_mass)/tot))

if __name__=='__main__':
    tg=sys.argv[1] if len(sys.argv)>1 else 'ekk'
    print("=== config A_base ===")
    diagnose(tg,n_iter=10,wpow=3.0,gap=-3.0,relpow=0.0,smooth=False)
