import numpy as np

VOWELS={
 'i':(2,4,0),'y':(2,4,1),'ɨ':(1,4,0),'ʉ':(1,4,1),'ɯ':(0,4,0),'u':(0,4,1),
 'ɪ':(2,3,0),'ʏ':(2,3,1),'ʊ':(0,3,1),
 'e':(2,3,0),'ø':(2,3,1),'ɘ':(1,3,0),'ɵ':(1,3,1),'ɤ':(0,3,0),'o':(0,3,1),
 'ə':(1,2,0),
 'ɛ':(2,2,0),'œ':(2,2,1),'ɜ':(1,2,0),'ʌ':(0,2,0),'ɔ':(0,2,1),
 'æ':(2,1,0),'ɐ':(1,1,0),
 'a':(2,0,0),'ɶ':(2,0,1),'ɑ':(0,0,0),'ɒ':(0,0,1),
}
# consonant: (place 0..9 bilabial..glottal, manner 0stop1affr2fric3nasal4trill/liquid5approx, voice)
CONS={
 'p':(0,0,0),'b':(0,0,1),'t':(2,0,0),'d':(2,0,1),'c':(6,0,0),'ɟ':(6,0,1),
 'k':(7,0,0),'g':(7,0,1),'q':(8,0,0),'ɢ':(8,0,1),'ʔ':(9,0,0),
 'ts':(2,1,0),'dz':(2,1,1),'tʃ':(3,1,0),'dʒ':(3,1,1),'tɕ':(4,1,0),'dʑ':(4,1,1),
 'cç':(6,1,0),'ɟʝ':(6,1,1),
 'f':(1,2,0),'v':(1,2,1),'β':(0,2,1),'ɸ':(0,2,0),'θ':(2,2,0),'ð':(2,2,1),
 's':(2,2,0),'z':(2,2,1),'ʃ':(3,2,0),'ʒ':(3,2,1),'ɕ':(4,2,0),'ʑ':(4,2,1),
 'ç':(6,2,0),'ʝ':(6,2,1),'x':(7,2,0),'ɣ':(7,2,1),'χ':(8,2,0),'ʁ':(8,2,1),
 'h':(9,2,0),'ɦ':(9,2,1),'ɬ':(2,2,0),'ɮ':(2,2,1),
 'm':(0,3,1),'ɱ':(1,3,1),'n':(2,3,1),'ɲ':(6,3,1),'ŋ':(7,3,1),'ɴ':(8,3,1),
 'r':(2,4,1),'ɾ':(2,4,1),'ʀ':(8,4,1),'l':(2,4,1),'ʎ':(6,4,1),'ʟ':(7,4,1),
 'j':(6,5,1),'w':(7,5,1),'ʋ':(1,5,1),'ɹ':(2,5,1),'ɰ':(7,5,1),'ɥ':(6,5,1),'ʝ':(6,2,1),
}

def strip_mods(seg):
    long_=0; pal=0; lab=0; asp=0; vless=0
    base=seg
    for d,flag in [('ː','long'),('ˑ','long'),('ʲ','pal'),('ʷ','lab'),('ʰ','asp'),('̥','vless'),('̊','vless'),('ʼ','ej')]:
        if d in base:
            base=base.replace(d,'')
            if flag=='long': long_=1
            elif flag=='pal': pal=1
            elif flag=='lab': lab=1
            elif flag=='asp': asp=1
            elif flag=='vless': vless=1
    # combining marks / accents
    for d in ['́','̀','̄','̂','̃','̈','̪','̻','̠','̬','ʰ','ⁿ']:
        base=base.replace(d,'')
    base=base.replace('́','').replace('̀','')
    return base,long_,pal,lab,asp,vless

def feat_vec(seg):
    base,long_,pal,lab,asp,vless=strip_mods(seg)
    v=np.zeros(14,dtype=np.float32)
    # decide vowel vs consonant
    # diphthong: 2+ vowel chars
    vchars=[c for c in base if c in VOWELS]
    if base in CONS:
        place,manner,voice=CONS[base]
        v[0]=1.0  # is-consonant
        v[1]=place/9.0
        v[2]=manner/5.0
        v[3]=voice
    elif base in VOWELS:
        bk,ht,rd=VOWELS[base]
        v[0]=0.0
        v[4]=1.0  # is-vowel
        v[5]=bk/2.0; v[6]=ht/4.0; v[7]=rd
    elif len(vchars)>=1:
        bk,ht,rd=VOWELS[vchars[0]]
        v[0]=0.0; v[4]=1.0; v[5]=bk/2.0; v[6]=ht/4.0; v[7]=rd
        v[13]=1.0  # diphthong flag
    elif len(base)>=1 and base[0] in CONS:
        place,manner,voice=CONS[base[0]]
        v[0]=1.0; v[1]=place/9.0; v[2]=manner/5.0; v[3]=voice
    else:
        # unknown -> neutral
        v[0]=0.5
    v[8]=long_; v[9]=pal; v[10]=lab; v[11]=asp; v[12]=vless
    return v

# feature weights for distance (emphasize class strongly)
WEIGHTS=np.array([3.0,1.6,1.6,0.9,3.0,1.3,1.3,0.7,0.6,0.7,0.5,0.4,0.5,0.4],dtype=np.float32)

def build_kernel(seg_vocab, tau=0.35, self_boost=2.0):
    F=np.stack([feat_vec(s) for s in seg_vocab])  # S x 14
    W=WEIGHTS
    S=len(seg_vocab)
    # weighted euclidean distance
    FW=F*np.sqrt(W)[None,:]
    sq=(FW*FW).sum(1)
    d2=sq[:,None]+sq[None,:]-2*FW@FW.T
    d2=np.maximum(d2,0.0)
    K=np.exp(-np.sqrt(d2)/tau)
    np.fill_diagonal(K, K.diagonal()+self_boost)
    K=K/K.sum(1,keepdims=True)
    return K.astype(np.float64)

if __name__=='__main__':
    import exp,collections
    segc=collections.Counter()
    for lg in exp.URAL:
        for _,segs in exp.by_lang[lg]:
            for s in segs: segc[s]+=1
    seg_vocab=sorted(segc)
    K=build_kernel(seg_vocab)
    idx={s:i for i,s in enumerate(seg_vocab)}
    for probe in ['k','t','a','i','tː','kʲ','s','u']:
        if probe in idx:
            row=K[idx[probe]]
            top=np.argsort(-row)[:6]
            print(probe,'~',[(seg_vocab[j],round(float(row[j]),3)) for j in top])
