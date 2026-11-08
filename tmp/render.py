import fitz
d = fitz.open('/home/user/rulb/tim_20261105/full_20261105/build/manuscript_v4.pdf')
print('pages', d.page_count)
for i, p in enumerate(d):
    p.get_pixmap(dpi=105).save('/tmp/pg%d.png' % (i+1))
print('rendered')