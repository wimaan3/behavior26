import json, hashlib, numpy as np
AG_SENTINEL=np.float32(100123456.0); AG_LEN=19
def get_uuid(n): return int(hashlib.md5(n.encode()).hexdigest(),16)%10**8
SYSREG=float(np.float32(get_uuid('system_registry')))
def build(sf):
    REG=sf['state']['registry']['object_registry']
    return REG,{float(np.float32(get_uuid(n))):n for n in REG}
def slen(o):
    if isinstance(o,dict): return sum(slen(v) for v in o.values())
    if isinstance(o,list): return sum(slen(x) for x in o) if (o and isinstance(o[0],(list,dict))) else len(o)
    return 0 if o is None else 1
def fill(o,v,i):
    if isinstance(o,dict):
        out={}
        for k,val in o.items(): out[k],i=fill(val,v,i)
        return out,i
    if isinstance(o,list):
        if o and isinstance(o[0],(list,dict)):
            out=[]
            for x in o: r,i=fill(x,v,i); out.append(r)
            return out,i
        return [float(x) for x in v[i:i+len(o)]],i+len(o)
    if o is None: return None,i
    return float(v[i]),i+1
def decode(v,REG,U2N):
    out={'scene_pos':[float(x) for x in v[:3]],'scene_ori':[float(x) for x in v[3:7]],
         'objects':{},'grasps':[],'errors':[],'order':[]}
    i=7; nreg=int(round(float(v[i]))); i+=1
    for _ in range(nreg):
        ru=float(v[i]); n=int(round(float(v[i+1]))); i+=2
        if ru==SYSREG:
            out['n_systems']=n
            if n: out['errors'].append('non-empty system_registry')
            continue
        out['n_objects']=n
        for _o in range(n):
            uu=float(v[i]); nm=U2N.get(uu)
            if nm is None: out['errors'].append(f'bad uuid@{i}'); return out
            rec,_=fill(REG[nm],v,i+1); out['objects'][nm]={'_off':i,**rec}; out['order'].append(nm)
            i+=1+slen(REG[nm])
    while i<len(v) and np.float32(v[i])==AG_SENTINEL:
        r=v[i:i+AG_LEN]
        out['grasps'].append({'active':float(r[1]),'obj':U2N.get(float(r[2]),f'?{float(r[2]):.0f}'),
                              'joint_type':float(r[3]),'local_pos':[float(x) for x in r[4:7]],
                              'quat':[float(x) for x in r[7:11]],'a':[float(x) for x in r[11:14]],
                              'b':[float(x) for x in r[14:17]],'tail':[float(x) for x in r[17:19]]})
        i+=AG_LEN
    if i!=len(v): out['errors'].append(f'consumed {i} != {len(v)}')
    return out

# ---------------- convenience API ----------------
import h5py
def load(path, demo='demo_0'):
    """Yields (scene_file_dict, REG, U2N, h5py file handle)."""
    f=h5py.File(path,'r'); sf=json.loads(f['data'].attrs['scene_file'])
    REG,U2N=build(sf); return sf,REG,U2N,f
def decode_episode(path, demo='demo_0', group=None):
    sf,REG,U2N,f=load(path,demo)
    g=f[group] if group else f['data/'+demo]
    ST=g['state']; SS=g['state_size'][:]
    out=[decode(ST[t][:SS[t]],REG,U2N) for t in range(len(SS))]
    f.close(); return sf,out
def carry_forward(frames, REG):
    """Fill absent objects with last-known value (falling back to scene_file init)."""
    import copy
    last={n:copy.deepcopy(REG[n]) for n in REG}; full=[]
    for fr in frames:
        last.update({n:v for n,v in fr['objects'].items()})
        full.append({n:copy.deepcopy(v) for n,v in last.items()})
    return full
