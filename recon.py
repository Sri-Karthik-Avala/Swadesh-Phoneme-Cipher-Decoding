import csv, collections
langs = {}
lang_fam = {}
lang_subfam = {}
lang_nwords = collections.Counter()
train_concepts = set()
seg_count = collections.Counter()
with open('train.csv', encoding='utf-8') as f:
    r = csv.DictReader(f)
    for row in r:
        lang=row['language']; fam=row['family']; sub=row['subfamily']; con=row['concept']; ipa=row['ipa']
        lang_fam[lang]=fam; lang_subfam[lang]=sub
        lang_nwords[lang]+=1
        train_concepts.add(con)
        for s in ipa.split():
            seg_count[s]+=1
print("Total distinct languages:", len(lang_fam))
print("Total distinct families:", len(set(lang_fam.values())))
print("Total distinct subfamilies:", len(set(lang_subfam.values())))
print("Total distinct train concepts:", len(train_concepts))
print("Total distinct IPA segments in train:", len(seg_count))
print()
fam_langs = collections.Counter(lang_fam.values())
print("Families and #languages:")
for fam,c in fam_langs.most_common():
    print("  %s: %d langs"%(fam,c))
print()
print("=== URALIC languages ===")
ural = [l for l in lang_fam if lang_fam[l]=='Uralic']
for l in sorted(ural):
    print("  %s  subfam=%s  nwords=%d"%(l,lang_subfam[l],lang_nwords[l]))
print("Uralic subfamilies:", collections.Counter(lang_subfam[l] for l in ural))

test_concepts=set()
tokens=collections.Counter()
tlens=[]
nrows=0
with open('test.csv',encoding='utf-8') as f:
    r=csv.DictReader(f)
    for row in r:
        nrows+=1
        test_concepts.add(row['concept'])
        toks=row['cipher'].split()
        tlens.append(len(toks))
        for t in toks: tokens[t]+=1
print()
print("=== TEST ===")
print("Test rows:", nrows)
print("Distinct test concepts:", len(test_concepts))
print("Distinct cipher tokens:", len(tokens))
print("Token freq (top 25):", tokens.most_common(25))
print("Word length: min%d max%d mean%.2f"%(min(tlens),max(tlens),sum(tlens)/len(tlens)))
print("Concept overlap test-in-train:", len(test_concepts & train_concepts), "of", len(test_concepts))
missing = test_concepts - train_concepts
print("Test concepts NOT in train:", len(missing))
print("Top train segments:", seg_count.most_common(40))
