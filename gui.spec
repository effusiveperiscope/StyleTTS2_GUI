# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_data_files

datas = []
datas += collect_data_files('gruut')


a = Analysis(
    ['gui.py'],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='styletts2',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='styletts2',
)

#import os
#import shutil
#shutil.copy2('config.yaml','dist/ponysorter_b/config.yaml')
#shutil.copy2('hashes.json','dist/ponysorter_b/hashes.json')
#shutil.copy2('numget.jpg','dist/ponysorter_b/numget.jpg')
#shutil.copy2('episodes_labels_index.json',
#    'dist/ponysorter_b/episodes_labels_index.json')
#os.makedirs('dist/ponysorter_b/in_audio', exist_ok=True)
#shutil.copytree('sup_audio','dist/ponysorter_b/sup_audio')