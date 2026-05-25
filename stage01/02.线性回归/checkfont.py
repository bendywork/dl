from matplotlib.font_manager import fontManager
for f in fontManager.ttflist:
    if 'Hei' in f.name or 'SC' in f.name or 'CN' in f.name or 'CJK' in f.name or 'Fang' in f.name:
        print(f.name, f.fname)