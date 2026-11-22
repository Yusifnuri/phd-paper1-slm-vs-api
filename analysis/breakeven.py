"""Break-even deployment volume Q* = C_train / (c_API - c_FT/u). Inputs: A5 05-feasibility.md (B77 N=1000, Qwen/Phi): 
run time 0.08-0.23 h; 5 runs (4 lr trials + final) => 0.4-1.15 h; inference 20-60 s per 1,000 items at saturation; Haiku/Luna k=32 request $0.000126-0.000169 (sync), x0.5 batch.
Rate band 3.6-6.0 $/h (SXM/HGX prices; no NVL price). Illustrative only; all inputs are A5's [U] ranges."""
import itertools
H=(0.4,1.15); INF=(20/3600/1000,60/3600/1000); API=(0.000126,0.000169)
print('mode  u    r    C_train case  Q*(items)')
for mode,fac in (('sync',1),('batch',.5)):
  for u in (1,.5,.2):
    row=[]
    for label,(h,hi,r,a) in {'best':(H[0],INF[0],3.6,API[1]),'central':(sum(H)/2,sum(INF)/2,4.8,sum(API)/2),'worst':(H[1],INF[1],6.0,API[0])}.items():
        C=h*r; cft=hi*r/u; ca=a*fac
        q=C/(ca-cft) if ca>cft else float('inf')
        row.append(f"{label}: C=${C:.1f} cFT=${cft:.6f} cAPI=${ca:.6f} Q*={'never' if q==float('inf') else format(q,',.0f')}")
    print(mode,u,' | '.join(row))
