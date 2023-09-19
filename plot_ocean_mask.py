import matplotlib.pyplot as plt


def plot_ocean_mask(ocean_mask):
    plt.figure(figsize=(12, 6))
    plt.imshow(ocean_mask, cmap="Blues_r", extent=(-180, 180, -90, 90), origin="upper")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Ocean Mask")
    plt.colorbar(label="0: Land, 1: Ocean")
    plt.show()


