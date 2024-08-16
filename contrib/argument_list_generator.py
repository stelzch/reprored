from itertools import product
dataset_sizes = [ 1371, 20364, 239763, 290718, 394684, 413434, 436077, 1240377, 1338678, 3011099, 21410970 ]
processor_counts = [24, 48, 96]
repetitions = [300]
ks = [3000]

arglist = list()
for (n,p,k,r) in product(dataset_sizes, processor_counts, repetitions, ks):
    arglist.append(f"{n},{p},{k},{r}")

print(" ".join(arglist))

