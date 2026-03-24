#!/usr/bin/env python
import pdb,sys,os
import json

f=open(sys.argv[1],'r')
lf=f.readlines()
f.close()
lf="".join(lf)
lf=lf[5:]
tjson=json.loads(lf)
pdb.set_trace()
with open("idrem_downloaded.json","w") as f:
	json.dump(tjson,f,indent=2)
