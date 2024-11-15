import json

# 載入統計數據
with open('training_stats.json', 'r') as f:
    stats = json.load(f)

# 更新 README
readme_template = f"""
# Snake AI Training Stats

- **Total Episodes**: {stats['total_episodes']}
- **Wall Hits**: {stats['wall_hits']}
- **Self Hits**: {stats['self_hits']}
- **Average Score**: {stats['avg_score']:.2f}
- **Maximum Score**: {stats['max_score']:.2f}
"""

with open('README.md', 'w') as f:
    f.write(readme_template)
