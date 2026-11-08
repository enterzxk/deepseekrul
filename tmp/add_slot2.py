from pathlib import Path
p = Path('src/candidates/dspt.py')
s = p.read_text()

old1 = '        self.sensor_score = nn.Linear(d_model, 1)'
new1 = old1 + '\n        self.slot_score = nn.Linear(d_model, 1)'
assert old1 in s, 'init anchor missing'
s = s.replace(old1, new1, 1)

old2 = '    def forward(self, x, mask=None, baseline=None, cycle=None):'
new2 = '''    def slot_pool(self, h, mask):
        B, L, D = h.shape
        valid = mask.bool()
        m = valid.sum(dim=1).clamp_min(1)
        slots = [masked_mean(h, mask)]
        idx = torch.arange(L, device=h.device).unsqueeze(0).expand(B, L)
        start = L - m.unsqueeze(1)
        pos = idx - start
        seg_id = (pos.clamp_min(0) * 4) // m.unsqueeze(1)
        for k in range(4):
            seg = valid & (seg_id == k)
            slots.append(masked_mean(h, seg.to(h.dtype)))
        S = torch.stack(slots, dim=1)
        w = torch.softmax(self.slot_score(S).squeeze(-1), dim=1)
        return (w.unsqueeze(-1) * S).sum(dim=1), w

    def forward(self, x, mask=None, baseline=None, cycle=None, mode='dual'):'''
assert old2 in s, 'forward anchor missing'
s = s.replace(old2, new2, 1)

old3 = '        y = self.reg_head(h_f)'
new3 = '''        if mode == 'slot':
            h_s, w_slots = self.slot_pool(h, mask)
            return {'prediction': self.reg_head(h_s), 'y_A': self.reg_head(h_A),
                    'y_B': self.reg_head(h_B), 'gate': g, 'slot_w': w_slots, 'embedding': h_s}
        y = self.reg_head(h_f)'''
assert old3 in s, 'pred anchor missing'
s = s.replace(old3, new3, 1)

p.write_text(s)
print('slot pooling added; slot occurrences:', s.count('slot'))