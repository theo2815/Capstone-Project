import sys,glob
from PIL import Image,ImageDraw
fs=sorted(glob.glob('stills/t-*.png'),key=lambda f:float(f[9:-4]))
W,H=800,450;cols=3;rows=(len(fs)+cols-1)//cols
S=Image.new('RGB',(W*cols,H*rows),'white')
for i,f in enumerate(fs):
    im=Image.open(f).convert('RGB').resize((W,H));d=ImageDraw.Draw(im);d.rectangle([0,0,90,28],fill='red');d.text((6,6),f[9:-4],fill='white')
    S.paste(im,((i%cols)*W,(i//cols)*H))
S.save('sheet.png')
