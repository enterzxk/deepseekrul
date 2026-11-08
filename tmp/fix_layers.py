from pathlib import Path
p=Path('src/candidates/dspt.py'); s=p.read_text()
old='''        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=4, dim_feedforward=2 * d_model,
            dropout=dropout, activation='gelu', batch_first=True, norm_first=True)
        self.layers = nn.ModuleList([layer for _ in range(num_layers)])'''
new='''        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model, nhead=4, dim_feedforward=2 * d_model,
                dropout=dropout, activation='gelu', batch_first=True, norm_first=True)
            for _ in range(num_layers)])'''
assert old in s, 'anchor'
s=s.replace(old,new,1)
p.write_text(s); print('fixed')