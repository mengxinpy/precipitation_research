import matplotlib.pyplot as plt
import numpy as np

# Define a simple self-affine fractal - The Sierpinski Carpet
def sierpinski_carpet(iterations, size=900):
    carpet = np.ones((size, size))
    for i in range(iterations):
        step = size // 3**i
        for x in range(0, size, step):
            for y in range(0, size, step):
                if (x // step) % 3 == 1 and (y // step) % 3 == 1:
                    carpet[y:y+step, x:x+step] = 0
    return carpet

# Generate and plot the Sierpinski Carpet
iterations = 4
carpet = sierpinski_carpet(iterations)

plt.imshow(carpet, cmap='binary')
plt.title(f'Sierpinski Carpet (Iterations: {iterations})')
plt.axis('off')
plt.show()
