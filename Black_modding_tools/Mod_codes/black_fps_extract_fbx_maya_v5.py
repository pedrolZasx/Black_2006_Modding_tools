#!/usr/bin/env python3
"""
black_fps_extract_fbx.py - export the first-person skeleton (joints) + skinned meshes of a Black (2006, Xbox)
chunk01_skeleton.bin (or a full bg1_xxx.db) to a BINARY FBX 7.4 file.  Pure python + numpy, no FBX SDK.
Keep it in the same folder as black_fps_model_replace_v6.py (v5 / v2 also work).

  python black_fps_extract_fbx.py chunk01_skeleton.bin out.fbx                    # skeleton + gun + arms, skinned
  python black_fps_extract_fbx.py chunk01_skeleton.bin out.fbx --part gun         # skeleton + gun only
  python black_fps_extract_fbx.py chunk01_skeleton.bin out.fbx --part skeleton    # joints only
  python black_fps_extract_fbx.py chunk01_skeleton.bin out.fbx --parent Pistol_Root=Root --parent Wrist_lft=Joint_wrist_twist

What is read from the chunk (all verified on the pistol chunk):
  * bone names      : pointer table at container_base+0xC20 (count at +0x5C)
  * bind pose       : 44 inverse-bind 4x4 matrices (row-vector, metres, last column forced to 0,0,0,1) at container_base+0x70
  * skin weights    : per vertex  bytes 0..3 = 4 x u8 weights, bytes 4..11 = 4 x u16 bone index * 3
  * hierarchy       : NOT stored in a plain parent table -> derived from the bone names (same hierarchy as the game's clip rig:
                      Wrist_lft and Joint_wrist_twist are siblings under Elbow_lft, weapon root hangs off Wrist_rt, weapon parts under it).
                      Override anything with --parent Child=Parent.  World bind poses are exact either way; only local values change.
Units: FBX is written in centimetres (metres * 100), Y-up, same as rig.fbx.
"""
import argparse, math, os, re, struct, sys, zlib
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
for _m in ('black_fps_model_replace_v6', 'black_fps_model_replace_v5', 'black_fps_model_replace_v2'):
    try:
        R = __import__(_m); break
    except ImportError:
        R = None
if R is None: raise SystemExit('[!] black_fps_model_replace_v6.py (or v5/v2) must be in the same folder')

# ============================================================ chunk reading
def unpack_normal(w):
    ix, iy, iz = w & 0x7FF, (w >> 11) & 0x7FF, (w >> 22) & 0x3FF
    if ix >= 0x400: ix -= 0x800
    if iy >= 0x400: iy -= 0x800
    if iz >= 0x200: iz -= 0x400
    n = np.array([ix / 1023.0, iy / 1023.0, iz / 511.0]); l = np.linalg.norm(n)
    return n / l if l > 0 else n

def load(path):
    full = open(path, 'rb').read()
    s, e = R.locate_chunk(full, path.lower().endswith('.db'))
    c1 = bytearray(full[s:e]); return c1, R.parse_chunk(c1)

def _skel_header_ok(c1, b):
    if b < 0 or b + 0x64 > len(c1): return False
    off_m, off_p, n, off_n = struct.unpack_from('<4I', c1, b + 0x54)
    return off_m == 0x70 and 8 < n < 128 and off_p == off_m + 64 * n and off_n == off_p + 4 * n and b + off_n + 4 * n <= len(c1)

def find_skeleton_base(c1, info):
    """Skeleton container base. Pistol/rifle chunks store it at 0x98; some (e.g. heavy, base 0x100) store 0 there,
    so fall back to scanning for the skeleton header (112, 64*n+112, n, +4*n)."""
    b = info['container_base']
    if _skel_header_ok(c1, b): return b
    for b in range(0, min(len(c1), 0x10000), 16):
        if _skel_header_ok(c1, b): return b
    raise SystemExit('[!] skeleton header not found in this chunk')

def read_skeleton(c1, info):
    base = find_skeleton_base(c1, info)
    off_m, off_p, n, off_n = struct.unpack_from('<4I', c1, base + 0x54)
    if not (0 < n < 256): raise SystemExit('[!] bone table not found where expected (container_base+0x54)')
    names = []
    for i in range(n):
        o = base + struct.unpack_from('<I', c1, base + off_n + 4 * i)[0]
        names.append(bytes(c1[o:c1.index(b'\0', o)]).decode('latin1'))
    world = []                                           # column-vector 4x4, metres
    for i in range(n):
        m = np.array(struct.unpack_from('<16f', c1, base + off_m + 64 * i), float).reshape(4, 4); m[:, 3] = (0, 0, 0, 1)
        world.append(np.linalg.inv(m).T)
    return names, world

FINGER = re.compile(r'^(Finger\d|Thumb)_(Top|Mid|Tip)_(lft|rt)$')
def derive_parents(names, world, overrides, weapon_parent='Wrist_rt', table_parents=None):
    nm = set(names); par = {}
    weapon_roots = [n for n in names if n.endswith('_Root') and n != 'Root']
    for n in names:
        p = None; m = FINGER.match(n)
        if n == 'World': p = None
        elif n == 'Extra': p = 'World'
        elif n == 'Root': p = 'Extra'
        elif n.startswith('Shoulder_'): p = 'Root'
        elif n.startswith('Elbow_'): p = 'Shoulder_' + n.split('_')[1]
        elif n in ('Joint_wrist_twist', 'Wrist_lft'): p = 'Elbow_lft'
        elif n == 'Wrist_rt': p = 'Elbow_rt'
        elif m:
            k, part, side = m.groups()
            p = f'Wrist_{side}' if part == 'Top' else f'{k}_{"Top" if part == "Mid" else "Mid"}_{side}'
        elif n in weapon_roots: p = weapon_parent
        elif n.split('_')[0] + '_Root' in nm: p = n.split('_')[0] + '_Root'
        elif weapon_roots and not n.startswith(('Wrist', 'Joint', 'Elbow', 'Shoulder')): p = weapon_roots[0]     # extra gun bones (Bullets, G_Slide, Flap_Bone, ...)
        if p is not None and p not in nm: p = None
        if p is None and n != 'World':                    # unknown bone: nearest earlier bone
            i = names.index(n); d = [np.linalg.norm(world[i][:3, 3] - world[j][:3, 3]) for j in range(i)]
            p = names[int(np.argmin(d))]; print(f'[!] no naming rule for {n}: parent guessed = {p} (use --parent {n}=...)')
        par[n] = p
    if table_parents:                                    # gun-part parents inferred from the weapon's clips (black_weapon_table.py)
        for c, p in table_parents.items():
            if c in par and c not in weapon_roots and p in nm and not (FINGER.match(c) or c in ('World', 'Extra', 'Root', 'Joint_wrist_twist') or c.startswith(('Shoulder_', 'Elbow_', 'Wrist_'))): par[c] = p
    for c, p in overrides.items():
        if c not in nm or (p not in nm and p != 'None'): raise SystemExit(f'[!] --parent {c}={p}: unknown bone name')
        par[c] = None if p == 'None' else p
    return par

def group_skinned_mesh(c1, info, g):
    vs = info['vstart']; slots = g['slots']; loc = {s: i for i, s in enumerate(slots)}
    P, T, N, W = [], [], [], []
    for s in slots:
        vo = vs + s * R.STRIDE
        P.append(struct.unpack_from('<3f', c1, vo))
        u, v = struct.unpack_from('<2f', c1, vo + R.UV_OFF); T.append((u, -v))
        N.append(unpack_normal(struct.unpack_from('<I', c1, vo + R.NRM_OFF)[0]))
        raw = bytes(c1[vo + 12:vo + 24]); w = raw[0:4]; idx = struct.unpack_from('<4H', raw, 4)
        acc = {}
        for k in range(4):
            if w[k]: acc[idx[k] // 3] = acc.get(idx[k] // 3, 0) + w[k]
        tot = float(sum(acc.values())) or 1.0
        W.append({b: v / tot for b, v in acc.items()})
    tris = [(loc[a], loc[b], loc[c]) for a, b, c in R.strip_to_tris(g['indices'], info['total_verts'])]
    return np.array(P, float), np.array(T, float), np.array(N, float), tris, W

# ============================================================ binary FBX writer
class Node:
    def __init__(self, name, props=(), children=()):
        self.name, self.props, self.children = name, list(props), list(children)
    def add(self, *nodes):
        self.children.extend(nodes); return self

def _prop(v):
    if isinstance(v, bool): return b'C' + struct.pack('<B', int(v))
    if isinstance(v, str): b = v.encode('utf8'); return b'S' + struct.pack('<I', len(b)) + b
    if isinstance(v, bytes): return b'R' + struct.pack('<I', len(v)) + v
    if isinstance(v, int):
        return (b'I' + struct.pack('<i', v)) if -2**31 <= v < 2**31 else b'L' + struct.pack('<q', v)
    if isinstance(v, float): return b'D' + struct.pack('<d', v)
    if isinstance(v, tuple):                              # ('L', value) forced int64, ('d'|'i'|'l', array)
        t, a = v
        if t == 'L': return b'L' + struct.pack('<q', a)
        dt = {'d': '<f8', 'i': '<i4', 'l': '<i8'}[t]; raw = np.asarray(a, dt).tobytes()
        return t.encode() + struct.pack('<III', len(raw) // np.dtype(dt).itemsize, 0, len(raw)) + raw
    raise TypeError(type(v))

def _ser(n, pos):
    props = b''.join(_prop(p) for p in n.props); name = n.name.encode()
    head_len = 13 + len(name); body_pos = pos + head_len + len(props)
    kids = b''
    for c in n.children:
        kids += _ser(c, body_pos + len(kids))
    if n.children: kids += bytes(13)
    end = body_pos + len(kids)
    return struct.pack('<IIIB', end, len(n.props), len(props), len(name)) + name + props + kids

FOOT_ID = bytes.fromhex('fabcab09d0c8d466b176fb831cf7267e')
FOOT_MAGIC = bytes.fromhex('f85a8c6adef5d97eece90ce3758f290b')
def write_fbx(path, top, version=7400):
    out = b'Kaydara FBX Binary  \x00\x1a\x00' + struct.pack('<I', version)
    for n in top: out += _ser(n, len(out))
    out += bytes(13) + FOOT_ID + bytes(4)
    pad = (-len(out)) % 16; out += bytes(pad if pad else 16)       # align to 16; if already aligned Autodesk/Blender add 16 (never 0)
    out += struct.pack('<I', version) + bytes(120) + FOOT_MAGIC
    open(path, 'wb').write(out)


# ---- ASCII FBX 7.4 writer (Maya's FBX SDK rejects the hand-written binary as "File is corrupted"; ASCII loads fine)
def _a_str(v):
    if '\x00\x01' in v: a, b = v.split('\x00\x01', 1); v = b + '::' + a          # binary "name\0\1Class" -> ascii "Class::name"
    return '"' + v.replace('"', '&quot;') + '"'
def _a_num(v):
    if isinstance(v, bool): return 'T' if v else 'F'
    if isinstance(v, float): return repr(float(v))
    return str(int(v))
def _a_prop(v):
    if isinstance(v, str): return _a_str(v)
    if isinstance(v, tuple) and v[0] == 'L': return str(int(v[1]))
    return _a_num(v)
def _a_node(n, ind, out):
    t = '\t' * ind; arr = [p for p in n.props if isinstance(p, tuple) and p[0] in ('d', 'i', 'l')]
    raw = [p for p in n.props if isinstance(p, bytes)]
    if raw: return                                                                 # FileId etc. are not needed in ascii
    if arr:
        a = np.asarray(arr[0][1]); items = [_a_num(float(x)) if a.dtype.kind == 'f' else str(int(x)) for x in a.reshape(-1)]
        out.append(f'{t}{n.name}: *{len(items)} {{'); out.append(f'{t}\ta: ' + (',\n' + t + '\t   ').join(','.join(items[i:i+32]) for i in range(0, len(items), 32))); out.append(f'{t}}}'); return
    head = f'{t}{n.name}: ' + ', '.join(_a_prop(p) for p in n.props)
    if n.children or n.name in ('References', 'Properties70'):
        out.append(head + ' {')
        for c in n.children: _a_node(c, ind + 1, out)
        out.append(t + '}')
    else: out.append(head)
def write_fbx_ascii(path, top, version=7400):
    out = ['; FBX 7.4.0 project file', '; ------------------------------------------------------------------', '']
    for n in top:
        if n.name in ('FileId', 'CreationTime'): continue
        _a_node(n, 0, out); out.append('')
    open(path, 'w', newline='\n').write('\n'.join(out))

# ---- tiny reader used only to verify what we wrote
def read_fbx_tree(path):
    d = open(path, 'rb').read(); assert d.startswith(b'Kaydara FBX Binary  \x00\x1a\x00'); p = 27
    def rd(p):
        end, npr, pl, nl = struct.unpack_from('<IIIB', d, p); q = p + 13
        if end == 0: return None, q
        name = d[q:q + nl].decode(); q += nl; props = []
        for _ in range(npr):
            t = chr(d[q]); q += 1
            if t == 'S': l = struct.unpack_from('<I', d, q)[0]; props.append(d[q+4:q+4+l].decode()); q += 4 + l
            elif t == 'R': l = struct.unpack_from('<I', d, q)[0]; props.append(d[q+4:q+4+l]); q += 4 + l
            elif t == 'I': props.append(struct.unpack_from('<i', d, q)[0]); q += 4
            elif t == 'L': props.append(struct.unpack_from('<q', d, q)[0]); q += 8
            elif t == 'D': props.append(struct.unpack_from('<d', d, q)[0]); q += 8
            elif t == 'C': props.append(d[q]); q += 1
            elif t in 'dil':
                cnt, enc, cl = struct.unpack_from('<III', d, q); q += 12
                props.append(np.frombuffer(d[q:q+cl], {'d': '<f8', 'i': '<i4', 'l': '<i8'}[t])); q += cl
            else: raise ValueError(t)
        assert q == p + 13 + nl + pl, (name, q, p + 13 + nl + pl)
        kids = []
        while q < end:
            k, q = rd(q)
            if k is None: break
            kids.append(k)
        assert q == end, (name, q, end)
        return (name, props, kids), end
    nodes = []
    while True:
        n, p = rd(p)
        if n is None: break
        nodes.append(n)
    return nodes

# ============================================================ scene assembly
def P70(*rows):
    return Node('Properties70', children=[Node('P', list(r)) for r in rows])

def mat_to_array(m): return np.asarray(m, float).T.reshape(16)             # column-major (FBX / Blender convention)

def euler_xyz(R3):
    """FBX default order: R = Rz*Ry*Rx -> degrees (x, y, z)"""
    sy = -R3[2, 0]; ry = math.asin(max(-1.0, min(1.0, sy)))
    if abs(sy) < 0.99999:
        rx = math.atan2(R3[2, 1], R3[2, 2]); rz = math.atan2(R3[1, 0], R3[0, 0])
    else:
        rx = math.atan2(-R3[1, 2], R3[1, 1]); rz = 0.0
    return tuple(math.degrees(a) for a in (rx, ry, rz))

def build(names, world, parents, meshes, scale):
    uid = [1000000]
    def nid(): uid[0] += 1; return uid[0]
    S = np.diag([scale, scale, scale, 1.0])
    Wc = [w.copy() for w in world]
    for w in Wc: w[:3, 3] *= scale                                           # cm
    ids = {n: nid() for n in names}; attr = {n: nid() for n in names}
    objs, conns = [], []
    for n in names:
        p = parents[n]
        loc = Wc[names.index(n)] if p is None else np.linalg.inv(Wc[names.index(p)]) @ Wc[names.index(n)]
        rx, ry, rz = euler_xyz(loc[:3, :3]); t = loc[:3, 3]
        objs.append(Node('NodeAttribute', [('L', attr[n]), n + '\x00\x01NodeAttribute', 'LimbNode'],
                         [P70(('Size', 'double', 'Number', '', 2.0 * scale / 10)), Node('TypeFlags', ['Skeleton'])]))
        objs.append(Node('Model', [('L', ids[n]), n + '\x00\x01Model', 'LimbNode'], [
            Node('Version', [232]),
            P70(('Lcl Translation', 'Lcl Translation', '', 'A', float(t[0]), float(t[1]), float(t[2])),
                ('Lcl Rotation', 'Lcl Rotation', '', 'A', rx, ry, rz),
                ('Lcl Scaling', 'Lcl Scaling', '', 'A', 1.0, 1.0, 1.0),
                ('DefaultAttributeIndex', 'int', 'Integer', '', 0),     # REQUIRED: without it the FBX SDK (Maya) ignores the joint attribute -> empty groups
                ('InheritType', 'enum', '', '', 1)),
            Node('MultiLayer', [0]), Node('MultiTake', [0]), Node('Shading', [True]), Node('Culling', ['CullingOff'])]))
        conns.append(Node('C', ['OO', ('L', attr[n]), ('L', ids[n])]))
        conns.append(Node('C', ['OO', ('L', ids[n]), ('L', ids[p] if p else 0)]))
    stats = []; mesh_ids = []
    for mname, (Pv, T, N, tris, W) in meshes:
        gid, mid, skid = nid(), nid(), nid(); mesh_ids.append((mname, mid))
        V = Pv * scale; nv = len(V)
        corners = np.array(tris, int).reshape(-1)                              # triangles
        pvi = corners.copy().reshape(-1, 3); pvi[:, 2] = ~pvi[:, 2]            # last index of each polygon is negated (xor -1)
        nrm = N[corners]; uvs = T[corners]
        objs.append(Node('Geometry', [('L', gid), mname + '\x00\x01Geometry', 'Mesh'], [
            P70(), Node('GeometryVersion', [124]),
            Node('Vertices', [('d', V.reshape(-1))]), Node('PolygonVertexIndex', [('i', pvi.reshape(-1))]),
            Node('LayerElementNormal', [0], [Node('Version', [101]), Node('Name', ['']), Node('MappingInformationType', ['ByPolygonVertex']),
                                             Node('ReferenceInformationType', ['Direct']), Node('Normals', [('d', nrm.reshape(-1))])]),
            Node('LayerElementUV', [0], [Node('Version', [101]), Node('Name', ['UVMap']), Node('MappingInformationType', ['ByPolygonVertex']),
                                         Node('ReferenceInformationType', ['IndexToDirect']), Node('UV', [('d', T.reshape(-1))]),
                                         Node('UVIndex', [('i', corners)])]),
            Node('Layer', [0], [Node('Version', [100]),
                                Node('LayerElement', children=[Node('Type', ['LayerElementNormal']), Node('TypedIndex', [0])]),
                                Node('LayerElement', children=[Node('Type', ['LayerElementUV']), Node('TypedIndex', [0])])])]))
        objs.append(Node('Model', [('L', mid), mname + '\x00\x01Model', 'Mesh'], [
            Node('Version', [232]), P70(('Lcl Translation', 'Lcl Translation', '', 'A', 0.0, 0.0, 0.0),
                                        ('DefaultAttributeIndex', 'int', 'Integer', '', 0),   # REQUIRED for the mesh shape to attach (see joints)
                                        ('InheritType', 'enum', '', '', 1)),
            Node('MultiLayer', [0]), Node('MultiTake', [0]), Node('Shading', [True]), Node('Culling', ['CullingOff'])]))
        conns += [Node('C', ['OO', ('L', gid), ('L', mid)]), Node('C', ['OO', ('L', mid), ('L', 0)])]
        objs.append(Node('Deformer', [('L', skid), mname + '\x00\x01Deformer', 'Skin'],
                         [Node('Version', [101]), Node('Link_DeformAcuracy', [50.0])]))
        conns.append(Node('C', ['OO', ('L', skid), ('L', gid)]))
        per_bone = {}
        for vi, w in enumerate(W):
            for b, wt in w.items(): per_bone.setdefault(b, []).append((vi, wt))
        for b, lst in sorted(per_bone.items()):
            cid = nid(); link = Wc[b]
            objs.append(Node('Deformer', [('L', cid), names[b] + '\x00\x01SubDeformer', 'Cluster'], [
                Node('Version', [100]), Node('UserData', ['', '']),
                Node('Indexes', [('i', [v for v, _ in lst])]), Node('Weights', [('d', [w for _, w in lst])]),
                Node('Transform', [('d', mat_to_array(np.linalg.inv(link)))]),   # Maya/Blender convention: Transform = inverse(TransformLink) (verified against a Blender export that works in Maya)
                Node('TransformLink', [('d', mat_to_array(link))]),
                Node('TransformAssociateModel', [('d', mat_to_array(np.eye(4)))])]))
            conns += [Node('C', ['OO', ('L', cid), ('L', skid)]), Node('C', ['OO', ('L', ids[names[b]]), ('L', cid)])]
        stats.append((mname, nv, len(tris), len(per_bone)))
    pid = nid(); pose_children = []
    for n in names: pose_children.append(Node('PoseNode', children=[Node('Node', [('L', ids[n])]), Node('Matrix', [('d', mat_to_array(Wc[names.index(n)]))])]))
    for mn, mid_ in mesh_ids: pose_children.append(Node('PoseNode', children=[Node('Node', [('L', mid_)]), Node('Matrix', [('d', mat_to_array(np.eye(4)))])]))
    objs.append(Node('Pose', [('L', pid), 'BindPose\x00\x01Pose', 'BindPose'],
                     [Node('Type', ['BindPose']), Node('Version', [100]), Node('NbPoseNodes', [len(pose_children)])] + pose_children))
    nm = len(meshes)
    top = [
        Node('FBXHeaderExtension', children=[Node('FBXHeaderVersion', [1003]), Node('FBXVersion', [7400]), Node('EncryptionType', [0]), Node('CreationTimeStamp', children=[Node('Version', [1000]), Node('Year', [2026]), Node('Month', [10]), Node('Day', [18]), Node('Hour', [0]), Node('Minute', [0]), Node('Second', [0]), Node('Millisecond', [0])]),
                                              Node('Creator', ['black_fps_extract_fbx.py'])]),
        Node('FileId', [bytes.fromhex('28b32aebb624ccc2bfc8b02aa92bfcf1')]),
        Node('CreationTime', ['1970-01-01 10:00:00:000']),
        Node('Creator', ['black_fps_extract_fbx_maya.py']),
        Node('GlobalSettings', children=[Node('Version', [1000]), P70(('UpAxis', 'int', 'Integer', '', 1), ('UpAxisSign', 'int', 'Integer', '', 1),
                ('FrontAxis', 'int', 'Integer', '', 2), ('FrontAxisSign', 'int', 'Integer', '', 1),
                ('CoordAxis', 'int', 'Integer', '', 0), ('CoordAxisSign', 'int', 'Integer', '', 1),
                ('UnitScaleFactor', 'double', 'Number', '', 1.0), ('OriginalUnitScaleFactor', 'double', 'Number', '', 1.0), ('TimeMode', 'enum', '', '', 6), ('CustomFrameRate', 'double', 'Number', '', 30.0))]),
        Node('Documents', children=[Node('Count', [1]), Node('Document', [('L', 1), 'Scene', 'Scene'], [P70(), Node('RootNode', [('L', 0)])])]),
        Node('References'),
        Node('Definitions', children=[Node('Version', [100]), Node('Count', [len(objs) + 1]),
            Node('ObjectType', ['GlobalSettings'], [Node('Count', [1])]),
            Node('ObjectType', ['Model'], [Node('Count', [len(names) + nm]), Node('PropertyTemplate', ['FbxNode'], [P70(
                ('InheritType', 'enum', '', '', 0), ('DefaultAttributeIndex', 'int', 'Integer', '', -1),
                ('Lcl Translation', 'Lcl Translation', '', 'A', 0.0, 0.0, 0.0), ('Lcl Rotation', 'Lcl Rotation', '', 'A', 0.0, 0.0, 0.0),
                ('Lcl Scaling', 'Lcl Scaling', '', 'A', 1.0, 1.0, 1.0), ('Visibility', 'Visibility', '', 'A', 1.0),
                ('Visibility Inheritance', 'Visibility Inheritance', '', '', 1))])]),
            Node('ObjectType', ['NodeAttribute'], [Node('Count', [len(names)])]),
            Node('ObjectType', ['Geometry'], [Node('Count', [nm])]),
            Node('ObjectType', ['Deformer'], [Node('Count', [sum(1 for o in objs if o.name == 'Deformer')])]),
            Node('ObjectType', ['Pose'], [Node('Count', [1])])]),
        Node('Objects', children=objs), Node('Connections', children=conns), Node('Takes', children=[Node('Current', [''])])]
    return top, stats

def verify(path, names, meshes):
    tree = {n[0]: n for n in read_fbx_tree(path)}
    objs = tree['Objects'][2]; models = [o for o in objs if o[0] == 'Model']
    assert len(models) == len(names) + len(meshes), 'model count'
    geos = [o for o in objs if o[0] == 'Geometry']; assert len(geos) == len(meshes)
    for g, (mn, (Pv, T, N, tris, W)) in zip(geos, meshes):
        k = {c[0]: c[1] for c in g[2]}
        assert len(k['Vertices'][0]) == 3 * len(Pv) and len(k['PolygonVertexIndex'][0]) == 3 * len(tris)
        pv = k['PolygonVertexIndex'][0]; assert (pv[2::3] < 0).all() and (pv[0::3] >= 0).all() and (pv[1::3] >= 0).all()
        assert max(max(pv[0::3]), max(pv[1::3]), max(~pv[2::3])) < len(Pv)
    cl = [o for o in objs if o[0] == 'Deformer' and o[1][2] == 'Cluster']
    for c in cl:
        k = {x[0]: x[1] for x in c[2]}; assert len(k['Indexes'][0]) == len(k['Weights'][0]) and len(k['TransformLink'][0]) == 16
    ids = {o[1][0] for o in objs}; ids.add(0)
    for c in tree['Connections'][2]: assert c[1][1] in ids and c[1][2] in ids, c
    return len(cl)

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('chunk'); ap.add_argument('out_fbx')
    ap.add_argument('--part', choices=('both', 'gun', 'arm', 'skeleton'), default='both')
    ap.add_argument('--parent', action='append', default=[], help='Child=Parent override (Parent may be None)')
    ap.add_argument('--format', choices=('ascii', 'binary'), default='ascii', help='ascii (default) loads in Maya; binary loads in Blender')
    ap.add_argument('--weapon-parent', default='Wrist_rt', help='parent of the <Weapon>_Root bone (default Wrist_rt; try Root)')
    ap.add_argument('--scale', type=float, default=100.0, help='metres -> FBX units (default 100 = cm)')
    a = ap.parse_args()
    c1, info = load(a.chunk)
    names, world = read_skeleton(c1, info)
    tp = None
    if a.chunk.lower().endswith('.db'):
        try:
            import black_weapon_table as _T; tp = _T.infer(a.chunk, verbose=False)['parents']
        except (SystemExit, Exception) as e: print(f'[!] weapon table not available ({e}); gun-part parents from names')
    parents = derive_parents(names, world, dict(x.split('=', 1) for x in a.parent), a.weapon_parent, tp)
    print(f'[*] {len(names)} bones; groups: ' + ', '.join(g['name'] for g in info['groups']))
    meshes = []
    if a.part != 'skeleton':
        for g in info['groups']:
            is_arm = 'arm' in g['name'].lower()
            if a.part == 'both' or (a.part == 'arm') == is_arm: meshes.append((g['name'], group_skinned_mesh(c1, info, g)))
    top, stats = build(names, world, parents, meshes, a.scale)
    chk = a.out_fbx + '.verify.tmp'; write_fbx(chk, top); ncl = verify(chk, names, meshes); os.remove(chk)      # structural self-check on a binary copy
    if a.format == 'ascii': write_fbx_ascii(a.out_fbx, top)
    else: write_fbx(a.out_fbx, top)
    for n, nv, nt, nb in stats: print(f'[+] {n}: {nv:,} vertices, {nt:,} triangles, skinned to {nb} bones')
    print(f'[*] wrote {a.out_fbx} ({os.path.getsize(a.out_fbx):,} bytes): {len(names)} joints, {len(meshes)} mesh(es), {ncl} skin clusters - structure re-read OK')
    roots = [n for n in names if parents[n] is None]
    print('[*] hierarchy roots:', roots, '| weapon/odd parents:', {n: parents[n] for n in names if n.split("_")[0] not in ("Finger1", "Finger2", "Finger3", "Finger4", "Thumb") and (n.endswith("_Root") or n.startswith(("Wrist", "Joint")))})

if __name__ == '__main__':
    main()
