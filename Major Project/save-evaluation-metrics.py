import pandas as pd
import matplotlib.pyplot as plt
import os

# Read CSV
csv_file = "tokenizer_evaluation.csv"  # change this to your CSV file
df = pd.read_csv(csv_file)

# Remove 'status' column if it exists
df = df.drop(columns=["Status", "Model_Folder"], errors="ignore")

# Extract base name (no extension) for naming PNG
base_name = os.path.splitext(os.path.basename(csv_file))[0]
output_png = f"{base_name}.png"

# Create table figure (size scales with content, but not excessive)
fig, ax = plt.subplots(figsize=(10, len(df) * 0.3 + 1))  # balanced width & height
ax.axis('off')  # hide axes

# Add table
table = ax.table(
    cellText=df.values,
    colLabels=df.columns,
    loc='center',
    cellLoc='center'
)

# Auto set column widths to content
for i, col in enumerate(df.columns):
    table.auto_set_column_width(i)

# Style table
table.auto_set_font_size(False)
table.set_fontsize(8)     # good readability
table.scale(1.4, 1.2)     # moderate spacing to avoid overlaps

# Save table as PNG
plt.savefig(output_png, dpi=300, bbox_inches='tight')
plt.close()

print(f"Table saved as {output_png}")
