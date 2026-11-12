import pymupdf
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import Normalize

# ============================================================
# Input PDFs
# ============================================================

pdf_files = [
    (
        "/scratch/aldo/aicon/src/aicon/drawer_tutorial/data/normal/metrics/success/ss_ab_heatmap.pdf",
        "Normal"
    ),
    (
        "/scratch/aldo/aicon/src/aicon/drawer_tutorial/data/disturbance/metrics/success/ss_ab_heatmap.pdf",
        "Disturbance"
    ),
    (
        "/scratch/aldo/aicon/src/aicon/drawer_tutorial/data/bad_prior/metrics/success/ss_ab_heatmap.pdf",
        "Bad_Prior"
    ),
    (
        "/scratch/aldo/aicon/src/aicon/drawer_tutorial/data/noise/metrics/success/ss_ab_heatmap.pdf",
        "Noisy"
    ),
]

output_file = (
    "/scratch/aldo/aicon/src/aicon/drawer_tutorial/"
    "Pairwise_Sweep/Evaluation_Scripts/pairwise_combined.pdf"
)


# ============================================================
# Open PDFs
# ============================================================

docs = []

for filename, caption in pdf_files:
    doc = pymupdf.open(filename)
    docs.append((doc, caption))


# ============================================================
# Create output PDF
# ============================================================

output = pymupdf.open()


# Page size: A4 landscape
page_width = 841.89
page_height = 595.28

page = output.new_page(
    width=page_width,
    height=page_height
)


# ============================================================
# Layout
# ============================================================

margin_left = 35
margin_right = 70
margin_top = 25
margin_bottom = 35

colorbar_width = 35
gap = 15

available_width = (
    page_width
    - margin_left
    - margin_right
    - colorbar_width
    - gap
)

available_height = (
    page_height
    - margin_top
    - margin_bottom
)

cell_width = (available_width - gap) / 2
cell_height = (available_height - gap) / 2


# ============================================================
# Insert the four PDF pages
# ============================================================

positions = [
    (margin_left, margin_top),                         # Normal
    (margin_left + cell_width + gap, margin_top),     # Disturbance
    (margin_left, margin_top + cell_height + gap),     # Bad Prior
    (margin_left + cell_width + gap,
     margin_top + cell_height + gap),                 # Noisy
]


for (doc, caption), (x, y) in zip(docs, positions):

    source_page = doc[0]

    rect = pymupdf.Rect(
        x,
        y,
        x + cell_width,
        y + cell_height
    )

    page.show_pdf_page(
        rect,
        doc,
        0
    )

    # Caption
    page.insert_text(
        pymupdf.Point(
            x + cell_width / 2 - 25,
            y + cell_height + 12
        ),
        caption,
        fontsize=16
    )


# ============================================================
# Create colorbar as vector graphics
# ============================================================

# Colorbar position
cbar_x = page_width - margin_right + 5
cbar_y = margin_top + 20
cbar_height = available_height - 40

# Number of segments
n_segments = 100

cmap = mpl.colormaps["viridis"]
norm = Normalize(vmin=0, vmax=3)

segment_height = cbar_height / n_segments

for i in range(n_segments):

    value = i / (n_segments - 1)
    color = cmap(value)

    # Convert matplotlib RGB [0,1] -> PDF RGB [0,1]
    rect = pymupdf.Rect(
        cbar_x,
        cbar_y + cbar_height - (i + 1) * segment_height,
        cbar_x + colorbar_width,
        cbar_y + cbar_height - i * segment_height
    )

    page.draw_rect(
        rect,
        color=None,
        fill=color[:3],
        width=0
    )


# ============================================================
# Colorbar ticks
# ============================================================

for value in [0, 1, 2, 3]:

    y = cbar_y + cbar_height * (1 - value / 3)

    # Tick
    page.draw_line(
        pymupdf.Point(cbar_x + colorbar_width, y),
        pymupdf.Point(cbar_x + colorbar_width + 5, y),
        color=(0, 0, 0),
        width=0.8
    )

    # Label
    page.insert_text(
        pymupdf.Point(
            cbar_x + colorbar_width + 8,
            y + 4
        ),
        str(value),
        fontsize=16
    )


# ============================================================
# Colorbar label
# ============================================================

page.insert_text(
    pymupdf.Point(
        cbar_x - 10,
        cbar_y - 15
    ),
    "SS_ab",
    fontsize=16
)


# ============================================================
# Save
# ============================================================

output.save(output_file)

output.close()

for doc, _ in docs:
    doc.close()

print()
print("Created:")
print(output_file)