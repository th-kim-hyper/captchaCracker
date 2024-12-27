# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_all

# TkinterDnD2 모듈의 모든 파일을 수집
datas, binaries, hiddenimports = collect_all('tkinterdnd2')

a = Analysis(
    ['win.py'],
    pathex=[],
    binaries=binaries,
    datas=datas + [
        ('model', 'model'),
        ('images', 'images'),
        ('assets', 'assets'),
        (r'C:\python\captchaCracker\tkdnd2.9.4', 'tkdnd2.9.4')  # tkdnd 라이브러리 경로 추가

    ],
    hiddenimports=hiddenimports,
    hookspath=['.'],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='win',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['hi.works.ico'],
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='win',
)
