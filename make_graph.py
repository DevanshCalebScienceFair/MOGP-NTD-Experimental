import os
import pandas as pd
import matplotlib.pyplot as plt

# 1. Read data safely
csv_path = "results/history.csv"
if not os.path.exists(csv_path):
    raise FileNotFoundError(f"Could not find {csv_path}.")

df = pd.read_csv(csv_path)

# Print the columns so we can see what's actually in there
print("📊 Columns found in your CSV:", df.columns.tolist())

# 2. Auto-detect the X and Y columns
x_col = next((col for col in ['evaluated', 'n_evaluated', 'molecules_evaluated', 'iteration', 'step'] if col in df.columns), None)
y_col = next((col for col in ['hypervolume', 'hv', 'Hypervolume'] if col in df.columns), None)

if not x_col or not y_col:
    print("❌ Couldn't automatically find the column names! Paste the 'Columns found' list above into the chat.")
else:
    print(f"✅ Plotting '{y_col}' vs '{x_col}'...")
    
    # 3. Set up the plot
    plt.figure(figsize=(10, 6))
    plt.plot(df[x_col], df[y_col], marker='o', linewidth=2.5, color='#1f77b4', markersize=8)

    # 4. Titles and Labels
    plt.title("AI Learning Curve: Hypervolume Growth Over Time", fontsize=16, fontweight='bold', pad=15)
    plt.xlabel("Number of Molecules Evaluated", fontsize=12, labelpad=10)
    plt.ylabel("Hypervolume (Front Quality)", fontsize=12, labelpad=10)
    plt.grid(True, linestyle='--', alpha=0.5)

    # 5. Save directly to your Mac Desktop
    desktop_path = os.path.expanduser("~/Desktop/science_fair_graph.png")
    plt.savefig(desktop_path, dpi=300, bbox_inches='tight')

    print("\n" + "="*60)
    print(f"🎉 SUCCESS! Graph saved to your desktop:\n👉 {desktop_path}")
    print("="*60 + "\n")