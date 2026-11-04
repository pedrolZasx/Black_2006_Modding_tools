"""
Black viewmodel - one-click control rig (tested in Maya 2027 / Python 3)

HOW TO USE
    Paste into the Script Editor (Python tab) and run  ->  a small window opens.
    Click  BUILD CONTROL RIG.  The 2D picker window opens automatically.   (Middle-mouse drag the code to a shelf for a permanent button.)

WHAT IT BUILDS (non-destructive: the skeleton/skin are only driven by constraints)
    CTRL_Global / CTRL_Root ........ move the whole viewmodel
    Arms (lft + rt) ................ FK circles, IK box + pole diamond, FK_IK switch on <side>_arm_settings
    Fingers ........................ one FK circle per finger joint (every Finger*/Thumb* joint under the wrists)
    Weapon roots ................... one control for every *_Root joint directly under Wrist_rt
                                     and one box control per child joint; finger roots are excluded

BUTTONS
    BUILD CONTROL RIG ............. build everything (one undo step)
    Bake to Skeleton + Remove Rig . bakes the animation onto the real joints, then deletes the rig
    Remove Rig .................... deletes the rig and puts the skeleton back to its pre-rig pose
"""
import json
import maya.cmds as cmds

RIG_GRP     = 'Rig_GRP'
ROOT_JOINT  = 'Root'
IGNORED_WRIST_ROOTS = {'Finger1_Top_rt', 'Finger2_Top_rt', 'Finger3_Top_rt', 'Finger4_Top_rt', 'Thumb_Top_rt'}
ARMS = [  # side, [shoulder, elbow, wrist], color index
    ('lft', ['Shoulder_lft', 'Elbow_lft', 'Wrist_lft'], 6),
    ('rt',  ['Shoulder_rt',  'Elbow_rt',  'Wrist_rt'],  13),
]
COL_CENTER, COL_PISTOL = 17, 14
TAG, REVTAG = '_rigCon', '_rigRev'
TR, SC, VIS = ['tx', 'ty', 'tz'], ['sx', 'sy', 'sz'], ['v']
CHANNELS = ['tx', 'ty', 'tz', 'rx', 'ry', 'rz']


# ------------------------------------------------------------------ small helpers
def _p(n):      return cmds.xform(n, q=True, ws=True, t=True)
def _sub(a, b): return [a[i] - b[i] for i in range(3)]
def _add(a, b): return [a[i] + b[i] for i in range(3)]
def _mul(a, s): return [x * s for x in a]
def _dot(a, b): return sum(a[i] * b[i] for i in range(3))
def _len(a):    return _dot(a, a) ** 0.5
def _norm(a):
    l = _len(a) or 1.0
    return [x / l for x in a]


def _kids(j):
    return cmds.listRelatives(j, children=True, type='joint') or []


def _bone_axis(j):
    """Local axis the bone points along (used as the circle normal)."""
    k = _kids(j)
    v = cmds.getAttr((k[0] if k else j) + '.translate')[0]
    i = max(range(3), key=lambda n: abs(v[n]))
    a = [0, 0, 0]
    a[i] = 1
    return a


def _bone_len(j):
    k = _kids(j)
    if k:
        return _len(_sub(_p(k[0]), _p(j)))
    par = cmds.listRelatives(j, parent=True)
    return _len(_sub(_p(j), _p(par[0]))) if par else 1.0


def _size(j, k, lo=0.6, hi=6.0):
    return max(lo, min(hi, _bone_len(j) * k))


_BOX = [(-1,-1,-1),(-1,-1,1),(-1,1,1),(-1,1,-1),(-1,-1,-1),(1,-1,-1),(1,1,-1),(-1,1,-1),
        (1,1,-1),(1,1,1),(-1,1,1),(1,1,1),(1,-1,1),(-1,-1,1),(1,-1,1),(1,-1,-1)]
_DIA = [(0,1,0),(1,0,0),(0,0,1),(0,1,0),(-1,0,0),(0,0,-1),(0,1,0),(0,0,1),(-1,0,0),
        (0,-1,0),(0,0,1),(1,0,0),(0,-1,0),(0,0,-1),(1,0,0)]
_CROSS = [(-1,0,0),(1,0,0),(0,0,0),(0,0,1),(0,0,-1),(0,0,0),(0,1,0),(0,-1,0)]


def _shape(kind, name, r, axis):
    if kind == 'circle':
        return cmds.circle(name=name, nr=axis, r=r, ch=False)[0]
    pts = {'box': _BOX, 'diamond': _DIA, 'cross': _CROSS}[kind]
    return cmds.curve(name=name, d=1, p=[[x * r, y * r, z * r] for x, y, z in pts])


def _style(node, color):
    for s in cmds.listRelatives(node, shapes=True) or []:
        cmds.setAttr(s + '.overrideEnabled', 1)
        cmds.setAttr(s + '.overrideColor', color)


def _lock(c, attrs):
    for a in attrs:
        cmds.setAttr('%s.%s' % (c, a), lock=True, keyable=False, channelBox=False)


def make_ctrl(name, kind, target, r, color, axis=(1, 0, 0), parent=None, pos=None, rot=True):
    """Curve + zeroed offset group. rot=False -> world-aligned control (position only)."""
    c = _shape(kind, name, r, axis)
    g = cmds.group(c, name=name + '_GRP')
    if pos is not None:
        cmds.xform(g, ws=True, t=pos)
    else:
        cmds.matchTransform(g, target, pos=True, rot=rot)
    if parent:
        g = cmds.parent(g, parent)[0]
    _style(c, color)
    return c, g


def _con(kind, drivers, driven):
    if isinstance(drivers, str):
        drivers = [drivers]
    fn = getattr(cmds, kind)
    return fn(*(list(drivers) + [driven]), maintainOffset=True, name=driven + TAG)[0]


def _reverse(src_attr, name):
    r = cmds.createNode('reverse', name=name + REVTAG)
    cmds.connectAttr(src_attr, r + '.inputX')
    return r


# ------------------------------------------------------------------ arm (FK / IK / switch)
def _build_arm(side, chain, color, root_ctrl, rig):
    sh, el, wr = chain
    sx = 1 if side == 'lft' else -1
    holder = cmds.group(empty=True, name='%s_arm_chains_GRP' % side, parent=root_ctrl)

    def dup(suffix):
        out, par = [], holder
        for j in chain:
            d = cmds.duplicate(j, parentOnly=True, name='%s_%s' % (j, suffix))[0]
            d = cmds.parent(d, par)[0]
            out.append(d)
            par = d
        return out
    fk, ik = dup('FK'), dup('IK')
    cmds.makeIdentity(ik[0], apply=True, rotate=True)   # rotations -> jointOrient (stable IK)
    cmds.setAttr(holder + '.visibility', 0)

    # FK controls (nested)
    fk_groups, par = [], root_ctrl
    for j, f in zip(chain, fk):
        c, g = make_ctrl('%s_FK_CTRL' % j, 'circle', j, _size(j, 0.22), color,
                         axis=_bone_axis(j), parent=par)
        _lock(c, TR + SC + VIS)
        _con('orientConstraint', c, f)
        fk_groups.append(g)
        par = c

    # IK control + pole vector
    ikc, ikg = make_ctrl('%s_IK_CTRL' % wr, 'box', wr, 3.5, color, parent=root_ctrl, rot=False)
    _lock(ikc, SC + VIS)
    h = cmds.ikHandle(sj=ik[0], ee=ik[2], sol='ikRPsolver', name='%s_ikHandle' % side)[0]
    cmds.parent(h, ikc)
    cmds.setAttr(h + '.visibility', 0)
    _con('orientConstraint', ikc, ik[2])

    ps, pe, pw = _p(sh), _p(el), _p(wr)
    line = _sub(pw, ps)
    t = _dot(_sub(pe, ps), line) / _dot(line, line)
    d = _sub(pe, _add(ps, _mul(line, t)))
    d = _norm(d) if _len(d) > 1e-3 else [0, -1, 0]
    pc, pg = make_ctrl('%s_pole_CTRL' % side, 'diamond', None, 2.0, color, parent=root_ctrl,
                       pos=_add(pe, _mul(d, _len(line) * 0.4)))
    _lock(pc, SC + VIS)
    cmds.poleVectorConstraint(pc, h, name='%s_pole%s' % (side, TAG))

    # FK/IK switch
    sc, sg = make_ctrl('%s_arm_settings' % side, 'cross', None, 2.5, color, parent=rig,
                       pos=_add(_p(wr), [sx * 10, 6, 0]))
    _con('parentConstraint', wr, sg)
    _lock(sc, TR + SC + ['rx', 'ry', 'rz'] + VIS)
    cmds.addAttr(sc, ln='FK_IK', at='double', min=0, max=1, dv=0, k=True)
    rev = _reverse(sc + '.FK_IK', '%s_arm' % side)
    for i in range(3):
        # no maintain-offset: FK/IK/bind are world-aligned right now.
        # interpType 2 = shortest (the default 'average' breaks when quaternion signs differ)
        con = cmds.orientConstraint(fk[i], ik[i], chain[i], name=chain[i] + TAG)[0]
        cmds.setAttr(con + '.interpType', 2)
        w = cmds.orientConstraint(con, q=True, weightAliasList=True)
        cmds.connectAttr(rev + '.outputX', '%s.%s' % (con, w[0]))
        cmds.connectAttr(sc + '.FK_IK', '%s.%s' % (con, w[1]))
    cmds.connectAttr(rev + '.outputX', fk_groups[0] + '.visibility')
    cmds.connectAttr(sc + '.FK_IK', ikg + '.visibility')
    cmds.connectAttr(sc + '.FK_IK', pg + '.visibility')


# ------------------------------------------------------------------ fingers
def _build_fingers(wr, color, rig):
    for kid in _kids(wr):
        if kid.endswith('_Root') or kid in IGNORED_WRIST_ROOTS:
            continue
        j = kid
        while j:
            c, g = make_ctrl(j + '_CTRL', 'circle', j, _size(j, 0.4, 0.5, 2.0), color,
                             axis=_bone_axis(j), parent=rig)
            _lock(c, TR + SC + VIS)
            _con('parentConstraint', cmds.listRelatives(j, parent=True)[0], g)
            _con('orientConstraint', c, j)
            k = _kids(j)
            j = k[0] if k else None


# ------------------------------------------------------------------ pistol
def _build_pistol(root_ctrl, hand, rig, weapon_root):
    ctrl_name = weapon_root + '_CTRL'
    pc, pg = make_ctrl(ctrl_name, 'box', weapon_root, 3.0, COL_PISTOL,
                       parent=rig, rot=False)
    _lock(pc, SC + VIS)
    # space switch: Root space (weight 1) <-> Hand space (Follow_Hand)
    con = cmds.parentConstraint(root_ctrl, pg, mo=True, name=pg + TAG)[0]
    cmds.parentConstraint(hand, pg, mo=True, weight=0)
    cmds.setAttr(con + '.interpType', 2)
    cmds.addAttr(pc, ln='Follow_Hand', at='double', min=0, max=1, dv=0, k=True)
    rev = _reverse(pc + '.Follow_Hand', weapon_root)
    w = cmds.parentConstraint(con, q=True, weightAliasList=True)
    cmds.connectAttr(rev + '.outputX', '%s.%s' % (con, w[0]))
    cmds.connectAttr(pc + '.Follow_Hand', '%s.%s' % (con, w[1]))
    _con('parentConstraint', pc, weapon_root)
    for part in cmds.listRelatives(weapon_root, allDescendents=True, type='joint') or []:
        c, g = make_ctrl(part + '_CTRL', 'box', part, 1.0, COL_PISTOL, parent=rig)
        _lock(c, SC + VIS)
        _con('parentConstraint', cmds.listRelatives(part, parent=True)[0], g)
        _con('parentConstraint', c, part)


def _wrist_roots(wr):
    return [j for j in _kids(wr)
            if j.endswith('_Root') and j not in IGNORED_WRIST_ROOTS]



# ------------------------------------------------------------------ public API
def build_rig():
    if cmds.objExists(RIG_GRP):
        cmds.warning('Rig already exists - press "Remove Rig" first.')
        return
    need = [ROOT_JOINT] + [j for _, ch, _ in ARMS for j in ch]
    miss = [j for j in need if not cmds.objExists(j)]
    if miss:
        cmds.error('Missing joints: %s' % miss)
    cmds.undoInfo(openChunk=True, chunkName='buildControlRig')
    try:
        joints = cmds.ls(type='joint')
        rest = dict((j, [cmds.getAttr('%s.%s' % (j, a)) for a in CHANNELS]) for j in joints)
        rig = cmds.group(empty=True, name=RIG_GRP)
        cmds.addAttr(rig, ln='restPose', dt='string')
        cmds.setAttr(rig + '.restPose', json.dumps(rest), type='string')
        R = max((_p(j)[0] ** 2 + _p(j)[2] ** 2) ** 0.5 for j in joints) * 1.15
        gc, gg = make_ctrl('CTRL_Global', 'circle', None, R, COL_CENTER, axis=(0, 1, 0),
                           parent=rig, pos=(0, 0, 0))
        rc, rg = make_ctrl('CTRL_Root', 'circle', ROOT_JOINT, R * 0.55, COL_CENTER,
                           axis=(0, 1, 0), parent=gc, rot=False)
        _lock(gc, SC + VIS)
        _lock(rc, SC + VIS)
        _con('parentConstraint', rc, ROOT_JOINT)
        for side, chain, color in ARMS:
            _build_arm(side, chain, color, rc, rig)
            _build_fingers(chain[2], color, rig)
        for weapon_root in _wrist_roots('Wrist_rt'):
            _build_pistol(rc, ARMS[1][1][2], rig, weapon_root)
        cmds.select(gc)
    finally:
        cmds.undoInfo(closeChunk=True)
    show_picker()
    print('Control rig built.')


def _rig_constraints():
    return [c for c in (cmds.ls('*' + TAG + '*') or []) if 'onstraint' in cmds.nodeType(c)]


def remove_rig(restore=True):
    """Delete the rig. restore=True puts the skeleton back to the pose it had when the rig was built."""
    rest = {}
    if cmds.objExists(RIG_GRP + '.restPose'):
        rest = json.loads(cmds.getAttr(RIG_GRP + '.restPose'))
    cmds.undoInfo(openChunk=True, chunkName='removeControlRig')
    try:
        for c in _rig_constraints():
            if cmds.objExists(c):
                cmds.delete(c)
        for n in cmds.ls('*' + REVTAG) or []:
            cmds.delete(n)
        if cmds.objExists(RIG_GRP):
            cmds.delete(RIG_GRP)
        if restore:
            for j, vals in rest.items():
                if not cmds.objExists(j):
                    continue
                for a, v in zip(CHANNELS, vals):
                    plug = '%s.%s' % (j, a)
                    if not cmds.listConnections(plug, s=True, d=False):
                        try:
                            cmds.setAttr(plug, v[0][0] if isinstance(v, list) else v)
                        except RuntimeError:
                            pass
    finally:
        cmds.undoInfo(closeChunk=True)
    _close_picker()
    print('Control rig removed.')


def bake_to_skeleton():
    """Bake the animation onto the real joints, then remove the rig."""
    todo = {}
    for c in _rig_constraints():
        par = cmds.listRelatives(c, parent=True)
        if par and cmds.nodeType(par[0]) == 'joint':
            attrs = ['rx', 'ry', 'rz']
            if cmds.nodeType(c) == 'parentConstraint':
                attrs = ['tx', 'ty', 'tz'] + attrs
            todo.setdefault(par[0], attrs)
    if not todo:
        cmds.warning('Nothing to bake.')
        return
    t0 = cmds.playbackOptions(q=True, min=True)
    t1 = cmds.playbackOptions(q=True, max=True)
    for j, attrs in todo.items():
        cmds.bakeResults(j, attribute=attrs, time=(t0, t1), simulation=True, sampleBy=1,
                         preserveOutsideKeys=True)
    remove_rig(restore=False)
    print('Baked %d joints (%d-%d).' % (len(todo), t0, t1))


# ------------------------------------------------------------------ 2D picker
PICKER_WS = 'BlackRigPickerWS'
PICK_W, PICK_H = 410, 356


def _pick(name):
    """Click = select, Shift+click = add, Ctrl+click = toggle."""
    if not cmds.objExists(name):
        cmds.warning('Control not found: %s' % name)
        return
    m = cmds.getModifiers()
    if m & 4:
        cmds.select(name, toggle=True)
    elif m & 1:
        cmds.select(name, add=True)
    else:
        cmds.select(name, replace=True)


def _finger_label(j):
    return 'Th' if j.startswith('Thumb') else j[len('Finger'):].split('_')[0]


def _picker_buttons():
    """(control, label, x, y, w, h, rgb) - edit the numbers here to move buttons around."""
    CEN, PIS = (0.85, 0.72, 0.2), (0.3, 0.65, 0.4)
    cols = {'lft': (0.25, 0.45, 0.85), 'rt': (0.85, 0.3, 0.3)}
    B = [('CTRL_Global', 'GLOBAL', 145, 6, 120, 22, CEN),
         ('CTRL_Root', 'ROOT', 145, 32, 120, 22, CEN)]
    for (side, chain, _), x0 in zip(ARMS, (20, 240)):
        sh, el, wr = chain
        col = cols[side]
        B += [('%s_arm_settings' % side, '%s  FK / IK' % side[0].upper(), x0, 62, 150, 20, col),
              ('%s_IK_CTRL' % wr, 'IK hand', x0, 88, 72, 24, col),
              ('%s_pole_CTRL' % side, 'pole', x0 + 78, 88, 72, 24, col),
              ('%s_FK_CTRL' % sh, 'shoulder', x0, 120, 150, 24, col),
              ('%s_FK_CTRL' % el, 'elbow', x0, 148, 150, 24, col),
              ('%s_FK_CTRL' % wr, 'wrist', x0, 176, 150, 24, col)]
        fingers = [k for k in _kids(wr) if not k.endswith('_Root') and k not in IGNORED_WRIST_ROOTS]
        if side == 'rt':
            fingers.reverse()                      # thumb on the inner side
        for ci, top in enumerate(fingers):
            j, row = top, 0
            while j:
                B.append((j + '_CTRL', _finger_label(top) if row == 0 else '',
                          x0 + 30 * ci, 210 + 26 * row, 28, 24, col))
                k = _kids(j)
                j = k[0] if k else None
                row += 1
    weapon_roots = _wrist_roots('Wrist_rt') if cmds.objExists('Wrist_rt') else []
    for ri, weapon_root in enumerate(weapon_roots):
        B.append((weapon_root + '_CTRL', weapon_root[:12], 145, 296 + ri * 27, 120, 24, PIS))
        parts = cmds.listRelatives(weapon_root, allDescendents=True, type='joint') or []
        w = min(96, (PICK_W - 10) // max(len(parts), 1) - 3)
        for i, part in enumerate(parts):
            B.append((part + '_CTRL', part[:9], 5 + i * (w + 3), 324 + ri * 27, w, 24, PIS))
    return B


def _reset_selected():
    for n in cmds.ls(selection=True):
        for a in CHANNELS:
            plug = '%s.%s' % (n, a)
            if cmds.objExists(plug) and cmds.getAttr(plug, settable=True):
                cmds.setAttr(plug, 0)


def _select_all_ctrls():
    cmds.select(cmds.ls('*_CTRL', '*_arm_settings', type='transform'), replace=True)


def _picker_ui():
    cmds.columnLayout(adjustableColumn=True)
    form = cmds.formLayout(width=PICK_W, height=PICK_H)
    for n, label, x, y, w, h, col in _picker_buttons():
        b = cmds.button(label=label, width=w, height=h, backgroundColor=col,
                        annotation=n, command=lambda *_, n=n: _pick(n))
        cmds.formLayout(form, e=True, attachForm=[(b, 'top', y), (b, 'left', x)])
    cmds.setParent('..')
    cmds.separator(height=8)
    for label, attr in (('Left  FK/IK', 'lft_arm_settings.FK_IK'),
                        ('Right FK/IK', 'rt_arm_settings.FK_IK')):
        if cmds.objExists(attr):
            cmds.attrFieldSliderGrp(label=label, attribute=attr, minValue=0, maxValue=1,
                                    columnWidth3=(80, 45, 200))
    for weapon_root in (_wrist_roots('Wrist_rt') if cmds.objExists('Wrist_rt') else []):
        attr = weapon_root + '_CTRL.Follow_Hand'
        if cmds.objExists(attr):
            cmds.attrFieldSliderGrp(label=weapon_root + ' hand', attribute=attr, minValue=0, maxValue=1,
                                    columnWidth3=(80, 45, 200))
    cmds.separator(height=8)
    cmds.rowLayout(numberOfColumns=3, columnWidth3=(135, 135, 135))
    cmds.button(label='Reset selected', width=133, command=lambda *_: _reset_selected())
    cmds.button(label='Key selected', width=133, command=lambda *_: cmds.setKeyframe())
    cmds.button(label='Select all', width=133, command=lambda *_: _select_all_ctrls())
    cmds.setParent('..')


def _close_picker():
    if cmds.workspaceControl(PICKER_WS, q=True, exists=True):
        cmds.deleteUI(PICKER_WS, control=True)


def show_picker():
    """Floating, dockable 2D selector. Drag its tab onto the viewport edge to dock it."""
    _close_picker()
    cmds.workspaceControl(PICKER_WS, label='Black Rig Picker', floating=True, retain=False,
                          initialWidth=PICK_W + 20, initialHeight=PICK_H + 170)
    cmds.setParent(PICKER_WS)
    _picker_ui()


def show_ui():
    w = 'BlackControlRigWin'
    if cmds.window(w, exists=True):
        cmds.deleteUI(w)
    cmds.window(w, title='Black Control Rig', widthHeight=(230, 175), sizeable=False)
    cmds.columnLayout(adjustableColumn=True, rowSpacing=6)
    cmds.button(label='BUILD CONTROL RIG', height=44, backgroundColor=(0.28, 0.55, 0.35),
                command=lambda *_: build_rig())
    cmds.button(label='Bake to Skeleton + Remove Rig', height=28,
                command=lambda *_: bake_to_skeleton())
    cmds.button(label='Remove Rig', height=28, command=lambda *_: remove_rig())
    cmds.button(label='Open Picker', height=28, command=lambda *_: show_picker())
    cmds.showWindow(w)


if __name__ == '__main__':
    show_ui()
