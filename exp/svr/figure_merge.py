import os
from PIL import Image
import matplotlib.pyplot as plt

# Path where images are stored
img_dir = 'exp/svr/result'

# List of image filenames (adjust if necessary)
img_files = [
    "svr_ge_15m_agg.png", "svr_ge_15m_ind.png",
    "svr_ge_30m_agg.png", "svr_ge_30m_ind.png",
    "svr_ge_60m_agg.png", "svr_ge_60m_ind.png",
    "svr_nl_60m_agg.png", "svr_nl_60m_ind.png",
    "svr_uk_30m_agg.png", "svr_uk_30m_ind.png",
    "svr_uk_60m_agg.png", "svr_uk_60m_ind.png"
]

# Load images
images = [Image.open(os.path.join(img_dir, img)) for img in img_files]

# Define grid layout (e.g., 4 rows x 3 columns)
rows, cols = 4, 3
img_width, img_height = images[0].size
grid_width = cols * img_width
grid_height = rows * img_height

# Create a new blank image with a white background
grid_image = Image.new('RGB', (grid_width, grid_height), 'white')

# Paste images into grid
for index, img in enumerate(images):
    x = (index % cols) * img_width+1
    y = (index // cols) * img_height+1
    grid_image.paste(img, (x, y))

# Save the grid image
output_path = 'exp/svr/result/combined_grid.png'
grid_image.save(output_path)

# Display the combined image
plt.figure(figsize=(20, 8))
plt.imshow(grid_image)
plt.axis('off')
plt.show()
