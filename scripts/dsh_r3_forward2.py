t = open('src/candidates/model.py', encoding='utf-8', errors='ignore').read().splitlines()
for i in range(395, 445):
    print(f'{i+1:4d} {t[i]}')
