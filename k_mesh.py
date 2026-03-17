import numpy as np

data = np.loadtxt("chern_allbands.txt", dtype=str)
chern_kz = data["chern_kz"]   # shape (Nk, norbs)
Nk, norbs = chern_kz.shape

# sum over bands for each kz slice
sum_over_bands_per_kz = chern_kz.sum(axis=1)
# sum over all bands and all kz
total_sum = chern_kz.sum()

print("Sum over bands for each kz slice:")
for iz in range(Nk):
    print(f"kz slice {iz}: {sum_over_bands_per_kz[iz]}")

print("\nTotal sum over all bands and all kz:")
print(total_sum)
