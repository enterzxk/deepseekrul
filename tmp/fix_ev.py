from pathlib import Path
p=Path('scripts/run_dspt.py'); s=p.read_text()
old='''    def ev(mode):
        net.eval(); x,y,mk,b,c = VA; ps=[]
        with torch.no_grad():
            for i in range(0,len(x),512):
                o = net(x[i:i+512].to(dev), mask=mk[i:i+512].to(dev), baseline=b[i:i+512].to(dev), cycle=c[i:i+512].to(dev))'''
new='''    def ev(mode):
        net.eval(); ps=[]
        with torch.no_grad():
            for i in range(0,len(VA[0]),512):
                o = net(torch.from_numpy(VA[0][i:i+512]).to(dev),
                        mask=torch.from_numpy(VA[2][i:i+512]).to(dev),
                        baseline=torch.from_numpy(VA[3][i:i+512]).to(dev),
                        cycle=torch.from_numpy(VA[4][i:i+512]).to(dev))'''
assert old in s, 'anchor'
s=s.replace(old,new,1)
p.write_text(s); print('fixed ev')